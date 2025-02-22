'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

export default function Navbar() {
  const pathname = usePathname()

  const isActive = (path: string) => {
    return pathname === path
  }

  return (
    <nav className="bg-white shadow-sm">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex justify-between h-16">
          <div className="flex">
            <div className="flex-shrink-0 flex items-center">
              <Link href="/dashboard" className="text-2xl font-bold">
                Swaroop-ai
              </Link>
            </div>
            <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
              <Link
                href="/dashboard"
                className={`${
                  isActive('/dashboard')
                    ? 'border-blue-500 text-gray-900'
                    : 'border-transparent text-gray-500'
                } inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium`}
              >
                Size Recommendation
              </Link>
              <Link
                href="/dashboard/catalog"
                className={`${
                  isActive('/dashboard/catalog')
                    ? 'border-blue-500 text-gray-900'
                    : 'border-transparent text-gray-500'
                } inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium`}
              >
                Catalog
              </Link>
              <Link
                href="/dashboard/orders"
                className={`${
                  isActive('/dashboard/orders')
                    ? 'border-blue-500 text-gray-900'
                    : 'border-transparent text-gray-500'
                } inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium`}
              >
                Orders
              </Link>
            </div>
          </div>
        </div>
      </div>
    </nav>
  )
}