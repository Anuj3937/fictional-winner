import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Circle, ChevronLeft, ChevronRight, MoreHorizontal } from 'lucide-react';

interface OpenTab {
  path: string;
  name: string;
  language: string;
  content: string;
  isDirty: boolean;
  isReadonly?: boolean;
}

interface Props {
  tabs: OpenTab[];
  activeTab: string | null;
  onTabSelect: (path: string) => void;
  onTabClose: (path: string) => void;
  onCloseAll?: () => void;
  onCloseOthers?: (path: string) => void;
}

export default function TabManager({ 
  tabs, 
  activeTab, 
  onTabSelect, 
  onTabClose,
  onCloseAll,
  onCloseOthers 
}: Props) {
  const [showScrollButtons, setShowScrollButtons] = useState(false);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);
  const [showMenu, setShowMenu] = useState(false);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const checkScroll = () => {
      const container = scrollContainerRef.current;
      if (!container) return;

      const { scrollLeft, scrollWidth, clientWidth } = container;
      setShowScrollButtons(scrollWidth > clientWidth);
      setCanScrollLeft(scrollLeft > 0);
      setCanScrollRight(scrollLeft < scrollWidth - clientWidth - 1);
    };

    checkScroll();
    const container = scrollContainerRef.current;
    if (container) {
      container.addEventListener('scroll', checkScroll);
      const resizeObserver = new ResizeObserver(checkScroll);
      resizeObserver.observe(container);
      
      return () => {
        container.removeEventListener('scroll', checkScroll);
        resizeObserver.disconnect();
      };
    }
  }, [tabs]);

  // Auto-scroll to active tab
  useEffect(() => {
    if (activeTab && scrollContainerRef.current) {
      const activeTabElement = scrollContainerRef.current.querySelector(
        `[data-tab-path="${activeTab}"]`
      ) as HTMLElement;
      
      if (activeTabElement) {
        activeTabElement.scrollIntoView({
          behavior: 'smooth',
          block: 'nearest',
          inline: 'center'
        });
      }
    }
  }, [activeTab]);

  const scrollTabs = (direction: 'left' | 'right') => {
    const container = scrollContainerRef.current;
    if (!container) return;

    const scrollAmount = 200;
    const newScrollLeft = direction === 'left' 
      ? container.scrollLeft - scrollAmount
      : container.scrollLeft + scrollAmount;
    
    container.scrollTo({
      left: newScrollLeft,
      behavior: 'smooth'
    });
  };

  const getLanguageColor = (language: string) => {
    const colorMap: Record<string, string> = {
      'python': 'bg-green-500',
      'javascript': 'bg-yellow-500',
      'typescript': 'bg-blue-500',
      'json': 'bg-orange-500',
      'markdown': 'bg-gray-500',
      'yaml': 'bg-purple-500',
      'html': 'bg-orange-600',
      'css': 'bg-blue-400',
      'sql': 'bg-blue-700',
      'dockerfile': 'bg-indigo-500',
    };
    return colorMap[language] || 'bg-gray-400';
  };

  const handleCloseAll = () => {
    onCloseAll?.();
    setShowMenu(false);
  };

  const handleCloseOthers = () => {
    if (activeTab) {
      onCloseOthers?.(activeTab);
    }
    setShowMenu(false);
  };

  if (tabs.length === 0) {
    return null;
  }

  return (
    <div className="flex bg-gray-100 dark:bg-dark-700 border-b border-gray-200 dark:border-dark-600 relative">
      {/* Left scroll button */}
      {showScrollButtons && canScrollLeft && (
        <button
          onClick={() => scrollTabs('left')}
          className="flex-shrink-0 w-8 h-10 flex items-center justify-center hover:bg-gray-200 dark:hover:bg-dark-600 border-r border-gray-200 dark:border-dark-600"
        >
          <ChevronLeft className="w-4 h-4" />
        </button>
      )}

      {/* Tabs container */}
      <div 
        ref={scrollContainerRef}
        className="flex-1 flex overflow-x-auto scrollbar-hide"
        style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
      >
        <AnimatePresence>
          {tabs.map((tab) => (
            <motion.div
              key={tab.path}
              data-tab-path={tab.path}
              initial={{ opacity: 0, width: 0 }}
              animate={{ opacity: 1, width: 'auto' }}
              exit={{ opacity: 0, width: 0 }}
              transition={{ duration: 0.2 }}
              className={`flex items-center gap-2 px-4 py-2.5 border-r border-gray-200 dark:border-dark-600 cursor-pointer min-w-0 max-w-60 group flex-shrink-0 ${
                activeTab === tab.path
                  ? 'bg-white dark:bg-dark-800 text-gray-900 dark:text-gray-100 shadow-sm'
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-dark-600'
              }`}
              onClick={() => onTabSelect(tab.path)}
            >
              {/* Language indicator */}
              <div className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${getLanguageColor(tab.language)}`} />
              
              {/* Tab content */}
              <div className="flex items-center gap-2 min-w-0 flex-1">
                <span className="text-sm font-medium truncate">
                  {tab.name}
                </span>
                
                {/* Readonly indicator */}
                {tab.isReadonly && (
                  <span className="text-xs bg-gray-200 dark:bg-gray-600 text-gray-500 dark:text-gray-400 px-1.5 py-0.5 rounded">
                    RO
                  </span>
                )}
              </div>
              
              {/* Status indicators */}
              <div className="flex items-center gap-1 flex-shrink-0">
                {/* Dirty indicator */}
                {tab.isDirty && (
                  <Circle className="w-2 h-2 text-blue-500 fill-current" />
                )}
                
                {/* Close button */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onTabClose(tab.path);
                  }}
                  className={`p-1 hover:bg-gray-200 dark:hover:bg-dark-500 rounded transition-opacity ${
                    activeTab === tab.path ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
                  }`}
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Right scroll button */}
      {showScrollButtons && canScrollRight && (
        <button
          onClick={() => scrollTabs('right')}
          className="flex-shrink-0 w-8 h-10 flex items-center justify-center hover:bg-gray-200 dark:hover:bg-dark-600 border-l border-gray-200 dark:border-dark-600"
        >
          <ChevronRight className="w-4 h-4" />
        </button>
      )}

      {/* More menu */}
      <div className="relative">
        <button
          onClick={() => setShowMenu(!showMenu)}
          className="flex-shrink-0 w-8 h-10 flex items-center justify-center hover:bg-gray-200 dark:hover:bg-dark-600 border-l border-gray-200 dark:border-dark-600"
        >
          <MoreHorizontal className="w-4 h-4" />
        </button>

        <AnimatePresence>
          {showMenu && (
            <>
              <div 
                className="fixed inset-0 z-40"
                onClick={() => setShowMenu(false)}
              />
              <motion.div
                ref={menuRef}
                initial={{ opacity: 0, scale: 0.95, y: -10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95, y: -10 }}
                className="absolute top-full right-0 z-50 bg-white dark:bg-dark-800 border border-gray-200 dark:border-dark-700 rounded-lg shadow-lg py-2 min-w-[160px] mt-1"
              >
                <button
                  onClick={handleCloseOthers}
                  disabled={tabs.length <= 1}
                  className="w-full px-4 py-2 text-sm text-left hover:bg-gray-100 dark:hover:bg-dark-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Close Others
                </button>
                <button
                  onClick={handleCloseAll}
                  className="w-full px-4 py-2 text-sm text-left hover:bg-gray-100 dark:hover:bg-dark-700"
                >
                  Close All
                </button>
                
                <div className="border-t border-gray-200 dark:border-dark-600 my-2" />
                
                <div className="px-4 py-2">
                  <div className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Open Tabs ({tabs.length})
                  </div>
                  <div className="max-h-48 overflow-auto space-y-1">
                    {tabs.map((tab) => (
                      <button
                        key={tab.path}
                        onClick={() => {
                          onTabSelect(tab.path);
                          setShowMenu(false);
                        }}
                        className={`w-full text-left px-2 py-1 text-xs rounded flex items-center gap-2 ${
                          activeTab === tab.path
                            ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                            : 'hover:bg-gray-100 dark:hover:bg-dark-700'
                        }`}
                      >
                        <div className={`w-2 h-2 rounded-full ${getLanguageColor(tab.language)}`} />
                        <span className="truncate">{tab.name}</span>
                        {tab.isDirty && (
                          <Circle className="w-1.5 h-1.5 text-blue-500 fill-current flex-shrink-0" />
                        )}
                      </button>
                    ))}
                  </div>
                </div>
              </motion.div>
            </>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
