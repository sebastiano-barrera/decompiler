use std::{ffi::OsString, fs::File, io, marker::PhantomData, path::PathBuf};

use tracing::{event, Level};

pub type Key = u64;

pub trait Cache<T> {
    fn get_or_insert_with(&mut self, key: Key, insert_fn: &mut dyn FnMut() -> T) -> T;
}

pub struct FileSystemCache<T> {
    dir: PathBuf,
    size: usize,
    _ph: PhantomData<T>,
}

impl<T: for<'a> serde::Deserialize<'a> + serde::Serialize> FileSystemCache<T> {
    pub fn new(dir: PathBuf, size: usize) -> Self {
        if let Err(err) = std::fs::create_dir_all(dir.clone()) {
            event!(Level::ERROR, ?err, ?dir, "could not create cache directory");
        }

        FileSystemCache {
            dir,
            size,
            _ph: PhantomData,
        }
    }

    fn get(&self, key: Key) -> Option<T> {
        let Ok(_lock) = self.lock() else {
            event!(Level::ERROR, ?key, "could not acquire cache lock");
            return None;
        };

        let file_path = self.path_of_key(key);
        let Ok(file) = File::open(file_path.clone()) else {
            event!(Level::ERROR, ?file_path, "failed to open cache file");
            return None;
        };

        event!(Level::INFO, ?key, ?file_path, "cache hit");

        match rmp_serde::from_read(file) {
            Ok(value) => Some(value),
            Err(err) => {
                event!(Level::ERROR, ?file_path, ?err, "decoding error");
                None
            }
        }
    }

    fn put(&self, key: Key, value: &T) {
        let Ok(lock) = self.lock() else {
            event!(Level::ERROR, ?key, "could not acquire cache lock");
            return;
        };

        self.evict(&lock);
        let path = self.path_of_key(key);
        let mut file = match File::create(path.clone()) {
            Ok(file) => file,
            Err(err) => {
                event!(
                    Level::ERROR,
                    ?err,
                    ?key,
                    ?path,
                    "could not create file for value"
                );
                return;
            }
        };

        if let Err(err) = rmp_serde::encode::write(&mut file, value) {
            event!(
                Level::ERROR,
                ?err,
                ?key,
                ?path,
                "could not encode/write file for value"
            );
            return;
        }
    }

    fn path_of_key(&self, key: u64) -> PathBuf {
        use std::fmt::Write;

        let mut file_name = OsString::with_capacity(3 * 8);
        for b in key.to_le_bytes() {
            write!(file_name, "{:02x}-", b).unwrap();
        }

        self.dir.join(&file_name)
    }

    #[cfg(feature = "file_lock")]
    fn lock(&self) -> io::Result<File> {
        let f = File::open(self.dir.join("cache.lock"))?;
        f.lock()?;
        Ok(f)
    }
    #[cfg(not(feature = "file_lock"))]
    fn lock(&self) -> io::Result<File> {
        File::open("/dev/null")
    }

    /// Delete the oldest file(s) until the number of entries in the cache directory
    /// is lower than the cache's size.
    fn evict(&self, _lock_file: &File) {
        loop {
            let read_dir = match std::fs::read_dir(self.dir.clone()) {
                Ok(read_dir) => read_dir,
                Err(err) => {
                    event!(
                        Level::ERROR,
                        dir = ?self.dir,
                        ?err,
                        "could not list items in cache dir"
                    );
                    return;
                }
            };

            let files: Vec<_> = read_dir
                .flatten()
                .filter(|entry| {
                    let Ok(file_type) = entry.file_type() else {
                        return false;
                    };
                    file_type.is_file()
                })
                .collect();

            if files.len() < self.size {
                break;
            }

            let oldest = files
                .iter()
                // .metadata() and .created(): can they really fail?
                .min_by_key(|entry| entry.metadata().unwrap().created().unwrap())
                .unwrap();

            if let Err(err) = std::fs::remove_file(oldest.path()) {
                event!(
                    Level::ERROR,
                    path = ?oldest,
                    ?err,
                    "could not delete old file from cache (bailing out to avoid runaway file creating)"
                );
                return;
            }
        }
    }
}

impl<T: for<'a> serde::Deserialize<'a> + serde::Serialize> Cache<T> for FileSystemCache<T> {
    fn get_or_insert_with(&mut self, key: Key, insert_fn: &mut dyn FnMut() -> T) -> T {
        match self.get(key) {
            Some(value) => value,
            None => {
                let value = insert_fn();
                self.put(key, &value);
                value
            }
        }
    }
}

pub struct NullCache<T> {
    _ph: PhantomData<T>,
}

impl<T> NullCache<T> {
    pub fn new() -> Self {
        NullCache { _ph: PhantomData }
    }
}

impl<T> Cache<T> for NullCache<T> {
    fn get_or_insert_with(&mut self, _: Key, insert_fn: &mut dyn FnMut() -> T) -> T {
        insert_fn()
    }
}
