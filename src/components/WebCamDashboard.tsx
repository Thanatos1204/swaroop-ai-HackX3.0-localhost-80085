/* eslint-disable @typescript-eslint/no-unused-vars */
'use client';

import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Camera, RefreshCw } from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = 'https://14b4-2409-40c0-5c-2496-7912-76d4-255e-2c33.ngrok-free.app';

interface StreamState {
  isActive: boolean;
  stage: 'front' | 'side';
  error: string | null;
}

export default function PoseDetectionViewer() {
  const [streamState, setStreamState] = useState<StreamState>({
    isActive: false,
    stage: 'front',
    error: null
  });
  const imgRef = useRef<HTMLImageElement>(null);

  const startStream = () => {
    try {
      setStreamState(prev => ({ ...prev, isActive: true, error: null }));
      if (imgRef.current) {
        imgRef.current.src = `${API_BASE_URL}/video_feed`;
      }
    } catch (error) {
      handleStreamError();
    }
  };

  const stopStream = () => {
    setStreamState(prev => ({ ...prev, isActive: false }));
    if (imgRef.current) {
      imgRef.current.src = '';
    }
  };

  const handleStreamError = () => {
    setStreamState(prev => ({
      ...prev,
      isActive: false,
      error: 'Failed to connect to camera feed'
    }));
  };

  const restartProcess = async () => {
    try {
      // Stop current stream
      stopStream();
      
      // Call restart endpoint
      await axios.post(`${API_BASE_URL}/restart`);
      
      // Reset state and start new stream
      setStreamState({
        isActive: false,
        stage: 'front',
        error: null
      });
      
      // Short delay before starting new stream
      setTimeout(startStream, 500);
    } catch (error) {
      setStreamState(prev => ({
        ...prev,
        error: 'Failed to restart process'
      }));
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopStream();
    };
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-4xl mx-auto bg-white rounded-2xl shadow-xl p-8"
      >
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-4">Pose Detection</h1>
          <p className="text-gray-600">
            Stand in front of the camera and follow the on-screen instructions.
          </p>
        </div>

        {/* Stream Container */}
        <div className="relative aspect-square bg-black rounded-xl overflow-hidden mb-8">
          {!streamState.isActive ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={startStream}
                className="flex items-center space-x-2 bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700"
              >
                <Camera className="w-5 h-5" />
                <span>Start Camera</span>
              </motion.button>
            </div>
          ) : (
            <div className="relative w-full h-full">
              <img 
                ref={imgRef}
                alt="Pose Detection Stream"
                className="w-full h-full object-cover"
                onError={handleStreamError}
              />
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={stopStream}
                className="absolute top-4 right-4 bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700"
              >
                Stop Camera
              </motion.button>
            </div>
          )}
        </div>

        {/* Error Display */}
        {streamState.error && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mb-4 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg flex items-center justify-between"
          >
            <span>{streamState.error}</span>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={startStream}
              className="text-sm bg-red-100 px-3 py-1 rounded hover:bg-red-200"
            >
              Retry
            </motion.button>
          </motion.div>
        )}

        {/* Controls */}
        <div className="flex items-center justify-between">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={restartProcess}
            className="flex items-center space-x-2 bg-gray-100 text-gray-700 px-6 py-3 rounded-lg hover:bg-gray-200"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Restart Process</span>
          </motion.button>

          {/* Stream Status */}
          {streamState.isActive && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center space-x-3"
            >
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-sm text-gray-600">Camera Active</span>
              </div>
              <div className="h-4 w-px bg-gray-300" />
              <div className="text-sm text-gray-600">
                Stage: {streamState.stage}
              </div>
            </motion.div>
          )}
        </div>
      </motion.div>
    </div>
  );
}