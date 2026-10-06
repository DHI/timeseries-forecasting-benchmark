import { useEffect, useState } from 'react';

export type JsonDataState<T> =
  { status: 'loading' } | { status: 'error'; error: Error } | { status: 'ready'; data: T };

const cache = new Map<string, Promise<unknown>>();

/** Test-only: the cache otherwise outlives any single fetch mock, since it's module-scoped
 * rather than tied to a component lifetime — call this between tests that install a fresh
 * `fetch` mock for the same URL. */
export function clearJsonDataCache(): void {
  cache.clear();
}

function fetchJson<T>(path: string): Promise<T> {
  const url = import.meta.env.BASE_URL + path;
  let promise = cache.get(url) as Promise<T> | undefined;
  if (!promise) {
    promise = fetch(url).then((res) => {
      if (!res.ok) {
        throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
      }
      return res.json() as Promise<T>;
    });
    cache.set(url, promise);
  }
  return promise;
}

/** Fetches a static JSON data file (relative to the site's base path) once and keeps
 * it in memory, mirroring the Python app's `functools.cache`d loaders. */
export function useJsonData<T>(path: string): JsonDataState<T> {
  const [state, setState] = useState<JsonDataState<T>>({ status: 'loading' });

  useEffect(() => {
    let cancelled = false;
    setState({ status: 'loading' });
    fetchJson<T>(path).then(
      (data) => {
        if (!cancelled) setState({ status: 'ready', data });
      },
      (error: Error) => {
        if (!cancelled) setState({ status: 'error', error });
      },
    );
    return () => {
      cancelled = true;
    };
  }, [path]);

  return state;
}
