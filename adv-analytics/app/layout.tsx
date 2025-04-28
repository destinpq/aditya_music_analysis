import './globals.css?v=20240512'
import './page.module.css'
import './custom-styles.css'
import './styles/dataset-pages.css'
import './styles/upload-form.css'
import './styles/force-override.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import Navbar from './components/Navbar'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'DESTIN PQ Analytics',
  description: 'Advanced YouTube Analytics by DESTIN PQ',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <Navbar />
        <div className="main-content">
          <main className="main">
            {children}
          </main>
        </div>
      </body>
    </html>
  )
}
