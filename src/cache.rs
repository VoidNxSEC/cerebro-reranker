//! Lock-free concurrent cache

use dashmap::DashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use blake3::Hasher;

/// Cache entry with TTL
struct CacheEntry<V> {
    value: V,
    expires_at: Instant,
}

/// Thread-safe cache with TTL
pub struct LockFreeCache<K, V> {
    map: Arc<DashMap<K, CacheEntry<V>>>,
    ttl: Duration,
}

impl<K, V> LockFreeCache<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    pub fn new(ttl: Duration) -> Self {
        Self {
            map: Arc::new(DashMap::new()),
            ttl,
        }
    }

    pub fn get(&self, key: &K) -> Option<V> {
        self.map.get(key).and_then(|entry| {
            if entry.expires_at > Instant::now() {
                Some(entry.value.clone())
            } else {
                // Expired - remove
                drop(entry);
                self.map.remove(key);
                None
            }
        })
    }

    pub fn set(&self, key: K, value: V) {
        let entry = CacheEntry {
            value,
            expires_at: Instant::now() + self.ttl,
        };
        self.map.insert(key, entry);
    }

    pub fn remove(&self, key: &K) -> Option<V> {
        self.map.remove(key).map(|(_, entry)| entry.value)
    }

    pub fn clear(&self) {
        self.map.clear();
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Cleanup expired entries
    pub fn cleanup(&self) {
        let now = Instant::now();
        self.map.retain(|_, entry| entry.expires_at > now);
    }
}

/// Fast hash function for cache keys
pub fn fast_hash(data: &[u8]) -> u64 {
    use xxhash_rust::xxh3::xxh3_64;
    xxh3_64(data)
}

/// Content-addressed hash (BLAKE3)
pub fn content_hash(data: &[u8]) -> String {
    let mut hasher = Hasher::new();
    hasher.update(data);
    let hash = hasher.finalize();
    hash.to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_cache_basic() {
        let cache = LockFreeCache::new(Duration::from_secs(1));

        cache.set("key1", "value1");
        assert_eq!(cache.get(&"key1"), Some("value1"));

        cache.remove(&"key1");
        assert_eq!(cache.get(&"key1"), None);
    }

    #[test]
    fn test_cache_expiry() {
        let cache = LockFreeCache::new(Duration::from_millis(100));

        cache.set("key1", "value1");
        assert_eq!(cache.get(&"key1"), Some("value1"));

        thread::sleep(Duration::from_millis(150));
        assert_eq!(cache.get(&"key1"), None);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;

        let cache = Arc::new(LockFreeCache::new(Duration::from_secs(10)));
        let mut handles = vec![];

        for i in 0..10 {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                for j in 0..100 {
                    let key = format!("key_{}_{}", i, j);
                    cache_clone.set(key.clone(), j);
                    assert_eq!(cache_clone.get(&key), Some(j));
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(cache.len(), 1000);
    }
}
