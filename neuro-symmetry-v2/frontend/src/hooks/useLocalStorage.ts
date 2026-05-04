import { useCallback, useEffect, useState } from "react";

/**
 * Sync a piece of React state with localStorage. Reads on mount, writes on
 * change, and listens for `storage` events from other tabs.
 *
 * Stores raw strings — `T extends string` keeps this dependency-free. For
 * structured data, JSON-stringify at the call site.
 */
export function useLocalStorage<T extends string>(
  key:          string,
  initialValue: T,
): [T, (v: T) => void] {
  const read = useCallback((): T => {
    try {
      const v = localStorage.getItem(key);
      return (v ?? initialValue) as T;
    } catch {
      return initialValue;
    }
  }, [key, initialValue]);

  const [value, setValue] = useState<T>(read);

  const write = useCallback((v: T) => {
    setValue(v);
    try { localStorage.setItem(key, v); } catch { /* quota / private mode */ }
  }, [key]);

  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === key) setValue((e.newValue ?? initialValue) as T);
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [key, initialValue]);

  return [value, write];
}
