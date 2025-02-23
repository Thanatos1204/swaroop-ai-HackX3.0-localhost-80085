/* eslint-disable @typescript-eslint/no-unused-vars */
'use client'

import { useState, useRef } from 'react'
import { Camera } from 'lucide-react'
import type { SizeRecommendation as SizeRecommendationType } from '@/types'

export default function SizeRecommendation() {
  const [height, setHeight] = useState('')
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [images, setImages] = useState<{ front?: File; side?: File }>({})
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [recommendation, setRecommendation] = useState<SizeRecommendationType | null>(null)

  const handleImageUpload = (type: 'front' | 'side') => (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setImages(prev => ({ ...prev, [type]: e.target.files![0] }))
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    // Add size recommendation logic here
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h2 className="text-2xl font-bold mb-6">Get Your Size Recommendation</h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label className="block text-sm font-medium text-gray-700">Height (cm)</label>
          <input
            type="number"
            value={height}
            onChange={(e) => setHeight(e.target.value)}
            className="mt-1 w-full px-3 py-2 border border-gray-300 rounded-md"
          />
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Front Photo</label>
            <div className="mt-1 flex items-center">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center px-4 py-2 border border-gray-300 rounded-md"
              >
                <Camera className="mr-2" />
                Upload Front Photo
              </button>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleImageUpload('front')}
                className="hidden"
                accept="image/*"
              />
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700">Side Photo</label>
            <div className="mt-1 flex items-center">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center px-4 py-2 border border-gray-300 rounded-md"
              >
                <Camera className="mr-2" />
                Upload Side Photo
              </button>
              <input
                type="file"
                onChange={handleImageUpload('side')}
                className="hidden"
                accept="image/*"
              />
            </div>
          </div>
        </div>

        <button
          type="submit"
          className="w-full bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700"
        >
          Get Recommendation
        </button>
      </form>

      {recommendation && (
        <div className="mt-8 p-6 bg-gray-50 rounded-lg">
          <h3 className="text-xl font-semibold mb-4">Your Recommended Size</h3>
          <div className="space-y-2">
            <p>Size: {recommendation.size}</p>
            <p>Fit: {recommendation.fit}</p>
            <p>Confidence: {recommendation.confidence}%</p>
          </div>
        </div>
      )}
    </div>
  )
}