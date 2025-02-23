/* eslint-disable @typescript-eslint/no-unused-vars */
'use client'

import { JSXElementConstructor, Key, ReactElement, ReactNode, ReactPortal, useState } from 'react'
import { Order } from '@/types';
import Image from 'next/image';

export default function OrdersPage() {
  const [orders, setOrders] = useState<Order[]>([])

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h2 className="text-2xl font-bold mb-6">Order History</h2>
      <div className="space-y-6">
        {orders.map(order => (
          <div key={order.id} className="border rounded-lg p-6">
            <div className="flex justify-between items-center mb-4">
              <p className="font-semibold">Order #{order.id}</p>
              <p className="text-gray-600">{order.date}</p>
            </div>
            <div className="space-y-4">
              {order.items.map((item: { id: Key; imageUrl: string; name: string; brand: string; size: string; fit: string; price: number }) => (
                <div key={item.id} className="flex items-center gap-4">
                  <Image src={item.imageUrl || '/default-image.jpg'} alt={String(item.name)} className="w-20 h-20 object-cover rounded" />
                  <div>
                    <p className="font-semibold">{item.name}</p>
                    <p className="text-gray-600">{item.brand}</p>
                    <p>Size: {item.size} | Fit: {item.fit}</p>
                  </div>
                  <p className="ml-auto font-bold">${item.price}</p>
                </div>
              ))}
            </div>
            <div className="mt-4 pt-4 border-t">
              <p className="text-right font-bold">Total: ${order.totalAmount}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
