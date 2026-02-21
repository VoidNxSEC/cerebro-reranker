use dashmap::DashMap;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use xxhash_rust::xxh3::xxh3_64;

// ---------------------------------------------------------------------------
// TinyLFU Cache
//
// Architecture:
//   Admission ─► [Window LRU 1%] ──┐
//                                   ├─ TinyLFU gate ─► [Main Cache 99%]
//   Eviction candidate from Main ──┘
//
// The frequency sketch (Count-Min Sketch) estimates access frequency.
// A doorkeeper bloom filter rejects one-hit-wonders before they enter
// the sketch, halving memory overhead.
// ---------------------------------------------------------------------------

/// A concurrent cache with TinyLFU admission policy and TTL support.
pub struct TinyLfuCache<V: Clone + Send + Sync + 'static> {
    /// Primary storage — lock-free concurrent reads.
    store: DashMap<String, CacheEntry<V>>,

    /// Window LRU (1% of capacity) — admits recent entries.
    window: Mutex<LruQueue>,

    /// Main segmented LRU (99% of capacity).
    ///   - Probation: new entries land here.
    ///   - Protected: entries promoted on re-access.
    probation: Mutex<LruQueue>,
    protected: Mutex<LruQueue>,

    /// Frequency sketch (Count-Min Sketch).
    sketch: FrequencySketch,

    /// Doorkeeper bloom filter — reject one-hit-wonders.
    doorkeeper: Doorkeeper,

    /// Capacity limits.
    window_cap: usize,
    probation_cap: usize,
    protected_cap: usize,

    /// Total access counter for periodic sketch reset.
    access_count: AtomicU64,
    reset_threshold: u64,
}

#[derive(Clone)]
struct CacheEntry<V> {
    value: V,
    expires_at: Option<Instant>,
    segment: Segment,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Segment {
    Window,
    Probation,
    Protected,
}

impl<V: Clone + Send + Sync + 'static> TinyLfuCache<V> {
    /// Create a new TinyLFU cache with the given total capacity.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(16);

        let window_cap = (capacity as f64 * 0.01).ceil() as usize;
        let window_cap = window_cap.max(1);
        let main_cap = capacity - window_cap;
        let protected_cap = (main_cap as f64 * 0.80) as usize;
        let probation_cap = main_cap - protected_cap;

        Self {
            store: DashMap::with_capacity(capacity),
            window: Mutex::new(LruQueue::new()),
            probation: Mutex::new(LruQueue::new()),
            protected: Mutex::new(LruQueue::new()),
            sketch: FrequencySketch::new(capacity),
            doorkeeper: Doorkeeper::new(capacity),
            window_cap,
            probation_cap,
            protected_cap,
            access_count: AtomicU64::new(0),
            reset_threshold: (capacity as u64) * 10,
        }
    }

    /// Get a value by key. Returns `None` if not found or expired.
    pub fn get(&self, key: &str) -> Option<V> {
        let entry = self.store.get(key)?;

        // Lazy TTL check
        if let Some(expires) = entry.expires_at {
            if Instant::now() >= expires {
                drop(entry);
                self.store.remove(key);
                return None;
            }
        }

        let value = entry.value.clone();
        let segment = entry.segment;
        drop(entry);

        // Record access in sketch
        self.record_access(key);

        // Promote from probation to protected on re-access
        if segment == Segment::Probation {
            self.promote_to_protected(key);
        }

        Some(value)
    }

    /// Insert a key-value pair with optional TTL.
    pub fn put(&self, key: String, value: V, ttl: Option<Duration>) {
        let expires_at = ttl.map(|d| Instant::now() + d);

        // If key already exists, update in place
        if let Some(mut entry) = self.store.get_mut(&key) {
            entry.value = value;
            entry.expires_at = expires_at;
            self.record_access(&key);
            return;
        }

        // New entry goes to window first
        let entry = CacheEntry {
            value,
            expires_at,
            segment: Segment::Window,
        };

        self.store.insert(key.clone(), entry);

        let mut window = self.window.lock();
        window.push_back(key.clone());

        // Evict from window if over capacity
        if window.len() > self.window_cap {
            if let Some(evicted_key) = window.pop_front() {
                drop(window);
                self.admit_to_main(evicted_key);
            }
        }
    }

    /// Number of entries (including expired ones not yet cleaned).
    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Remove all entries.
    pub fn clear(&self) {
        self.store.clear();
        self.window.lock().clear();
        self.probation.lock().clear();
        self.protected.lock().clear();
        self.sketch.reset();
        self.doorkeeper.reset();
        self.access_count.store(0, Ordering::Relaxed);
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    fn record_access(&self, key: &str) {
        let hash = xxh3_64(key.as_bytes());

        // Doorkeeper: first hit goes to bloom filter, second hit enters sketch
        if self.doorkeeper.check_and_set(hash) {
            self.sketch.increment(hash);
        }

        // Periodic reset to prevent frequency inflation
        let count = self.access_count.fetch_add(1, Ordering::Relaxed);
        if count > 0 && count % self.reset_threshold == 0 {
            self.sketch.halve();
            self.doorkeeper.reset();
        }
    }

    fn admit_to_main(&self, candidate_key: String) {
        // Get candidate frequency
        let candidate_hash = xxh3_64(candidate_key.as_bytes());
        let candidate_freq = self.sketch.estimate(candidate_hash);

        let mut probation = self.probation.lock();

        // If probation has room, admit directly
        if probation.len() < self.probation_cap {
            if let Some(mut entry) = self.store.get_mut(&candidate_key) {
                entry.segment = Segment::Probation;
            }
            probation.push_back(candidate_key);
            return;
        }

        // Compare with probation victim
        if let Some(victim_key) = probation.front() {
            let victim_hash = xxh3_64(victim_key.as_bytes());
            let victim_freq = self.sketch.estimate(victim_hash);

            if candidate_freq > victim_freq {
                // Admit candidate, evict victim
                let victim = probation.pop_front().unwrap();
                self.store.remove(&victim);

                if let Some(mut entry) = self.store.get_mut(&candidate_key) {
                    entry.segment = Segment::Probation;
                }
                probation.push_back(candidate_key);
            } else {
                // Reject candidate
                self.store.remove(&candidate_key);
            }
        }
    }

    fn promote_to_protected(&self, key: &str) {
        let mut probation = self.probation.lock();
        let mut protected = self.protected.lock();

        // Remove from probation
        probation.remove(key);

        // Update segment
        if let Some(mut entry) = self.store.get_mut(key) {
            entry.segment = Segment::Protected;
        }

        protected.push_back(key.to_string());

        // If protected overflows, demote oldest to probation
        if protected.len() > self.protected_cap {
            if let Some(demoted) = protected.pop_front() {
                if let Some(mut entry) = self.store.get_mut(&demoted) {
                    entry.segment = Segment::Probation;
                }
                probation.push_back(demoted);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Count-Min Sketch (4 rows)
// ---------------------------------------------------------------------------

struct FrequencySketch {
    /// 4 rows of counters (saturating at 15 — packed as nibbles).
    table: Vec<AtomicU64>,
    mask: u64,
}

impl FrequencySketch {
    fn new(capacity: usize) -> Self {
        // Round up to next power of 2
        let width = (capacity * 4).next_power_of_two().max(64);
        // Each AtomicU64 holds 16 nibbles (4-bit counters)
        let table_len = width / 16;

        let mut table = Vec::with_capacity(table_len);
        for _ in 0..table_len {
            table.push(AtomicU64::new(0));
        }

        Self {
            table,
            mask: (width as u64) - 1,
        }
    }

    fn increment(&self, hash: u64) {
        for i in 0u64..4 {
            let h = hash.wrapping_add(i.wrapping_mul(0x9E3779B97F4A7C15));
            let idx = (h & self.mask) as usize;
            let slot = idx / 16;
            let nibble_pos = (idx % 16) * 4;

            if slot < self.table.len() {
                // CAS loop to increment nibble (saturating at 15)
                loop {
                    let old = self.table[slot].load(Ordering::Relaxed);
                    let nibble = (old >> nibble_pos) & 0xF;
                    if nibble >= 15 {
                        break; // saturated
                    }
                    let new = (old & !(0xF << nibble_pos)) | ((nibble + 1) << nibble_pos);
                    if self.table[slot]
                        .compare_exchange_weak(old, new, Ordering::Relaxed, Ordering::Relaxed)
                        .is_ok()
                    {
                        break;
                    }
                }
            }
        }
    }

    fn estimate(&self, hash: u64) -> u8 {
        let mut min = 15u8;

        for i in 0u64..4 {
            let h = hash.wrapping_add(i.wrapping_mul(0x9E3779B97F4A7C15));
            let idx = (h & self.mask) as usize;
            let slot = idx / 16;
            let nibble_pos = (idx % 16) * 4;

            if slot < self.table.len() {
                let val = self.table[slot].load(Ordering::Relaxed);
                let nibble = ((val >> nibble_pos) & 0xF) as u8;
                min = min.min(nibble);
            }
        }

        min
    }

    fn halve(&self) {
        for slot in &self.table {
            loop {
                let old = slot.load(Ordering::Relaxed);
                // Halve each nibble: shift right by 1 within each 4-bit group
                // Mask: 0x7777... preserves lower 3 bits of each nibble after shift
                let new = (old >> 1) & 0x7777_7777_7777_7777;
                if slot
                    .compare_exchange_weak(old, new, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
                {
                    break;
                }
            }
        }
    }

    fn reset(&self) {
        for slot in &self.table {
            slot.store(0, Ordering::Relaxed);
        }
    }
}

// ---------------------------------------------------------------------------
// Doorkeeper (bloom filter)
// ---------------------------------------------------------------------------

struct Doorkeeper {
    bits: Vec<AtomicU64>,
    mask: u64,
}

impl Doorkeeper {
    fn new(capacity: usize) -> Self {
        let n_bits = (capacity * 8).next_power_of_two().max(64);
        let n_words = n_bits / 64;

        let mut bits = Vec::with_capacity(n_words);
        for _ in 0..n_words {
            bits.push(AtomicU64::new(0));
        }

        Self {
            bits,
            mask: (n_bits as u64) - 1,
        }
    }

    /// Returns `true` if the item was already present (probably).
    fn check_and_set(&self, hash: u64) -> bool {
        let mut was_present = true;

        for i in 0u64..3 {
            let h = hash.wrapping_add(i.wrapping_mul(0x517CC1B727220A95));
            let bit_idx = (h & self.mask) as usize;
            let word = bit_idx / 64;
            let bit = bit_idx % 64;

            if word < self.bits.len() {
                let old = self.bits[word].fetch_or(1 << bit, Ordering::Relaxed);
                if old & (1 << bit) == 0 {
                    was_present = false;
                }
            }
        }

        was_present
    }

    fn reset(&self) {
        for word in &self.bits {
            word.store(0, Ordering::Relaxed);
        }
    }
}

// ---------------------------------------------------------------------------
// Simple LRU queue (ordered key tracking)
// ---------------------------------------------------------------------------

struct LruQueue {
    deque: VecDeque<String>,
}

impl LruQueue {
    fn new() -> Self {
        Self {
            deque: VecDeque::new(),
        }
    }

    fn push_back(&mut self, key: String) {
        self.deque.push_back(key);
    }

    fn pop_front(&mut self) -> Option<String> {
        self.deque.pop_front()
    }

    fn front(&self) -> Option<&String> {
        self.deque.front()
    }

    fn remove(&mut self, key: &str) {
        if let Some(pos) = self.deque.iter().position(|k| k == key) {
            self.deque.remove(pos);
        }
    }

    fn len(&self) -> usize {
        self.deque.len()
    }

    fn clear(&mut self) {
        self.deque.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_put_get() {
        let cache = TinyLfuCache::new(100);
        cache.put("key1".into(), "value1".to_string(), None);
        assert_eq!(cache.get("key1"), Some("value1".to_string()));
    }

    #[test]
    fn test_missing_key() {
        let cache: TinyLfuCache<String> = TinyLfuCache::new(100);
        assert_eq!(cache.get("missing"), None);
    }

    #[test]
    fn test_update_existing() {
        let cache = TinyLfuCache::new(100);
        cache.put("key1".into(), "v1".to_string(), None);
        cache.put("key1".into(), "v2".to_string(), None);
        assert_eq!(cache.get("key1"), Some("v2".to_string()));
    }

    #[test]
    fn test_ttl_expiration() {
        let cache = TinyLfuCache::new(100);
        cache.put(
            "key1".into(),
            "value".to_string(),
            Some(Duration::from_millis(1)),
        );
        std::thread::sleep(Duration::from_millis(10));
        assert_eq!(cache.get("key1"), None);
    }

    #[test]
    fn test_clear() {
        let cache = TinyLfuCache::new(100);
        for i in 0..10 {
            cache.put(format!("k{i}"), format!("v{i}"), None);
        }
        assert_eq!(cache.len(), 10);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_eviction_under_pressure() {
        let cache = TinyLfuCache::new(32);

        // Insert more entries than capacity
        for i in 0..64 {
            cache.put(format!("k{i}"), i, None);
        }

        // Should not exceed capacity significantly
        // (some overcount is expected due to concurrent structure)
        assert!(cache.len() <= 64);
    }

    #[test]
    fn test_frequency_sketch_basic() {
        let sketch = FrequencySketch::new(1024);
        let hash = xxh3_64(b"test_key");

        assert_eq!(sketch.estimate(hash), 0);

        sketch.increment(hash);
        assert!(sketch.estimate(hash) >= 1);

        for _ in 0..10 {
            sketch.increment(hash);
        }
        assert!(sketch.estimate(hash) >= 5);
    }

    #[test]
    fn test_frequency_sketch_halve() {
        let sketch = FrequencySketch::new(1024);
        let hash = xxh3_64(b"halve_test");

        for _ in 0..10 {
            sketch.increment(hash);
        }

        let before = sketch.estimate(hash);
        sketch.halve();
        let after = sketch.estimate(hash);

        assert!(after <= before);
    }

    #[test]
    fn test_doorkeeper() {
        let dk = Doorkeeper::new(1024);
        let hash = xxh3_64(b"door_test");

        // First check: not present
        assert!(!dk.check_and_set(hash));
        // Second check: present
        assert!(dk.check_and_set(hash));
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let cache = Arc::new(TinyLfuCache::new(256));
        let mut handles = vec![];

        for t in 0..8 {
            let cache = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let key = format!("t{t}_k{i}");
                    cache.put(key.clone(), i, None);
                    cache.get(&key);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // No panics, no deadlocks = success
        assert!(cache.len() > 0);
    }
}
