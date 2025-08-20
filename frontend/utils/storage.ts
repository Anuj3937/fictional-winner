interface StorageUtils {
  setItem: (key: string, value: any) => boolean;
  getItem: <T>(key: string, defaultValue?: T) => T | null;
  removeItem: (key: string) => boolean;
  clear: () => boolean;
  getKeys: () => string[];
  getSize: () => number;
}

class LocalStorageUtils implements StorageUtils {
  private prefix = 'ml-playground-';

  private getFullKey(key: string): string {
    return this.prefix + key;
  }

  setItem(key: string, value: any): boolean {
    try {
      const serializedValue = JSON.stringify({
        data: value,
        timestamp: Date.now(),
        type: typeof value
      });
      localStorage.setItem(this.getFullKey(key), serializedValue);
      return true;
    } catch (error) {
      console.error('Failed to set localStorage item:', error);
      return false;
    }
  }

  getItem<T>(key: string, defaultValue?: T): T | null {
    try {
      const item = localStorage.getItem(this.getFullKey(key));
      if (!item) {
        return defaultValue || null;
      }

      const parsed = JSON.parse(item);
      return parsed.data as T;
    } catch (error) {
      console.error('Failed to get localStorage item:', error);
      return defaultValue || null;
    }
  }

  removeItem(key: string): boolean {
    try {
      localStorage.removeItem(this.getFullKey(key));
      return true;
    } catch (error) {
      console.error('Failed to remove localStorage item:', error);
      return false;
    }
  }

  clear(): boolean {
    try {
      const keys = this.getKeys();
      keys.forEach(key => {
        localStorage.removeItem(this.getFullKey(key));
      });
      return true;
    } catch (error) {
      console.error('Failed to clear localStorage:', error);
      return false;
    }
  }

  getKeys(): string[] {
    try {
      const keys: string[] = [];
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key?.startsWith(this.prefix)) {
          keys.push(key.substring(this.prefix.length));
        }
      }
      return keys;
    } catch (error) {
      console.error('Failed to get localStorage keys:', error);
      return [];
    }
  }

  getSize(): number {
    try {
      let total = 0;
      this.getKeys().forEach(key => {
        const item = localStorage.getItem(this.getFullKey(key));
        if (item) {
          total += item.length;
        }
      });
      return total;
    } catch (error) {
      console.error('Failed to calculate localStorage size:', error);
      return 0;
    }
  }

  // Additional utility methods
  isQuotaExceeded(): boolean {
    try {
      const testKey = 'test-quota';
      localStorage.setItem(testKey, 'test');
      localStorage.removeItem(testKey);
      return false;
    } catch (error) {
      return true;
    }
  }

  getItemWithExpiry<T>(key: string, defaultValue?: T): T | null {
    try {
      const item = localStorage.getItem(this.getFullKey(key));
      if (!item) {
        return defaultValue || null;
      }

      const parsed = JSON.parse(item);
      
      // Check if item has expiry and is expired
      if (parsed.expiry && Date.now() > parsed.expiry) {
        this.removeItem(key);
        return defaultValue || null;
      }

      return parsed.data as T;
    } catch (error) {
      console.error('Failed to get localStorage item with expiry:', error);
      return defaultValue || null;
    }
  }

  setItemWithExpiry(key: string, value: any, ttlMinutes: number): boolean {
    try {
      const expiry = Date.now() + (ttlMinutes * 60 * 1000);
      const serializedValue = JSON.stringify({
        data: value,
        timestamp: Date.now(),
        expiry,
        type: typeof value
      });
      localStorage.setItem(this.getFullKey(key), serializedValue);
      return true;
    } catch (error) {
      console.error('Failed to set localStorage item with expiry:', error);
      return false;
    }
  }
}

export const storageUtils = new LocalStorageUtils();

// Session storage utilities
export const sessionStorageUtils = {
  setItem: (key: string, value: any): boolean => {
    try {
      sessionStorage.setItem(`ml-playground-${key}`, JSON.stringify(value));
      return true;
    } catch (error) {
      console.error('Failed to set sessionStorage item:', error);
      return false;
    }
  },

  getItem: <T>(key: string, defaultValue?: T): T | null => {
    try {
      const item = sessionStorage.getItem(`ml-playground-${key}`);
      return item ? JSON.parse(item) : (defaultValue || null);
    } catch (error) {
      console.error('Failed to get sessionStorage item:', error);
      return defaultValue || null;
    }
  },

  removeItem: (key: string): boolean => {
    try {
      sessionStorage.removeItem(`ml-playground-${key}`);
      return true;
    } catch (error) {
      console.error('Failed to remove sessionStorage item:', error);
      return false;
    }
  },

  clear: (): boolean => {
    try {
      sessionStorage.clear();
      return true;
    } catch (error) {
      console.error('Failed to clear sessionStorage:', error);
      return false;
    }
  }
};
