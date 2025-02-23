/* eslint-disable @typescript-eslint/no-unused-vars */
'use client'

import { useState } from 'react'
import Image from 'next/image'
import { ClothingItem } from '@/types'

export default function CatalogPage() {
  const [items, setItems] = useState<ClothingItem[]>([])
  const [selectedSize, setSelectedSize] = useState('')
  const [selectedFit, setSelectedFit] = useState('')

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">Clothing Catalog</h2>
        <div className="flex gap-4">
          <select
            value={selectedSize}
            onChange={(e) => setSelectedSize(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md"
          >
            <option value="">All Sizes</option>
            {['S', 'M', 'L', 'XL', 'XXL'].map(size => (
              <option key={size} value={size}>{size}</option>
            ))}
          </select>
          <select
            value={selectedFit}
            onChange={(e) => setSelectedFit(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md"
          >
            <option value="">All Fits</option>
            {['Slim', 'Regular', 'Loose', 'Oversized'].map(fit => (
              <option key={fit} value={fit}>{fit}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-6">
        {items.map(item => (
          <div key={item.id} className="border rounded-lg overflow-hidden">
            <Image src={item.imageUrl} alt={item.name} className="w-full h-64 object-cover" />
            <div className="p-4">
              <h3 className="font-semibold">{item.name}</h3>
              <p className="text-gray-600">{item.brand}</p>
              <p>Size: {item.size}</p>
              <p>Fit: {item.fit}</p>
              <p className="font-bold mt-2">${item.price}</p>
              <button className="mt-2 w-full bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700">
                Add to Cart
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

