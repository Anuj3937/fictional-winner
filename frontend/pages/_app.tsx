import { useEffect, useState } from 'react';
import type { AppProps } from 'next/app';
import { Inter, JetBrains_Mono } from 'next/font/google';
import { Toaster } from 'react-hot-toast';
import { motion, AnimatePresence } from 'framer-motion';
import { ThemeProvider } from 'next-themes';
import Head from 'next/head';
import ErrorBoundary from '@/components/ui/ErrorBoundary';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import '@/styles/globals.css';

// Font configurations
const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
  display: 'swap',
});

export default function App({ Component, pageProps, router }: AppProps) {
  const [mounted, setMounted] = useState(false);

  // Fix hydration issues
  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <>
        <Head>
          <title>Loading - ML Automation Playground</title>
          <meta name="viewport" content="width=device-width, initial-scale=1" />
        </Head>
        <div className={`${inter.variable} ${jetbrainsMono.variable} font-sans`}>
          <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
            <LoadingSpinner 
              variant="brain" 
              size="xl" 
              text="Initializing AI-powered ML platform..."
            />
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <Head>
        <title>ML Automation Playground - AI-Powered Machine Learning</title>
        <meta 
          name="description" 
          content="Create complete ML solutions using natural language with our intelligent agent system" 
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="theme-color" content="#3B82F6" />
        <link rel="icon" href="/favicon.ico" />
        <link rel="manifest" href="/manifest.json" />
      </Head>

      <div className={`${inter.variable} ${jetbrainsMono.variable} font-sans antialiased`}>
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem={true}
          disableTransitionOnChange={false}
        >
          <ErrorBoundary>
            <AnimatePresence
              mode="wait"
              initial={false}
              onExitComplete={() => {
                if (typeof window !== 'undefined') {
                  window.scrollTo(0, 0);
                }
              }}
            >
              <motion.div
                key={router.asPath}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{
                  duration: 0.3,
                  ease: [0.4, 0.0, 0.2, 1]
                }}
              >
                <Component {...pageProps} />
              </motion.div>
            </AnimatePresence>

            <Toaster
              position="top-right"
              gutter={12}
              containerClassName="z-50"
              toastOptions={{
                duration: 4000,
                className: 'toast',
                style: {
                  borderRadius: '12px',
                  fontSize: '14px',
                  fontWeight: '500',
                  padding: '12px 16px',
                  maxWidth: '500px',
                },
                success: {
                  iconTheme: {
                    primary: '#10B981',
                    secondary: '#FFFFFF',
                  },
                },
                error: {
                  iconTheme: {
                    primary: '#EF4444',
                    secondary: '#FFFFFF',
                  },
                },
                loading: {
                  iconTheme: {
                    primary: '#3B82F6',
                    secondary: '#FFFFFF',
                  },
                },
              }}
            />
          </ErrorBoundary>
        </ThemeProvider>
      </div>
    </>
  );
}
