'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Button } from "@/components/ui/button";
import Image from "next/image";
import logo from "../../public/logo.png";

export default function Navbar() {
  const pathname = usePathname()

  const isActive = (path: string) => {
    return pathname === path
  }

  return (
    <nav className="bg-white shadow-sm w-full">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex justify-between h-16 items-center">
          <div className="flex items-center space-x-2">
            <Image src={logo} alt="Swaroop.ai Logo" width={50} height ={50}/>
            <Link href="/dashboard" className="text-2xl font-bold">
              Swaroop.ai
            </Link>
          </div>
          <Button variant="outline" className="text-1.5xl px-8 py-4 border-2 border-black rounded-lg">Login</Button>
        </div>
      </div>
    </nav>
  )
}