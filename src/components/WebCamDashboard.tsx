/* eslint-disable @typescript-eslint/no-unused-vars */
import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Camera, RefreshCw } from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

interface StreamState {
  isActive: boolean;
  stage: 'front' | 'side';
  error: string | null;
  frontImage: string | null;
  sideImage: string | null;
  isComplete: boolean;
}

export default function WebcamDashboard({ onImageCapture }: { onImageCapture: (image: File) => void }) {
  const [streamState, setStreamState] = useState<StreamState>({
    isActive: false,
    stage: 'front',
    error: null,
    frontImage: null,
    sideImage: null,
    isComplete: false
  });
  
  const imgRef = useRef<HTMLImageElement>(null);
  const [countdown, setCountdown] = useState<number | null>(null);

  const checkForImages = async () => {
    try {
      const frontResponse = await fetch(`${API_BASE_URL}/get_image/front`);
      const sideResponse = await fetch(`${API_BASE_URL}/get_image/side`);

      console.log(frontResponse)
      console.log(sideResponse)
      
      let frontCaptured = false;
      let sideCaptured = false;
  
      if (frontResponse.ok) {
        const frontBlob = await frontResponse.blob();
        const frontFile = new File([frontBlob], "front_pose.jpg", { type: "image/jpeg" });
        setStreamState(prev => ({
          ...prev,
          frontImage: URL.createObjectURL(frontBlob),
          stage: 'side'
        }));
        frontCaptured = true;
        onImageCapture(frontFile)
      }
  
      if (sideResponse.ok) {
        const sideBlob = await sideResponse.blob();
        setStreamState(prev => ({
          ...prev,
          sideImage: URL.createObjectURL(sideBlob),
          isComplete: true,
          isActive: false
        }));
        sideCaptured = true;
      }
  
      // If both images are captured, stop the stream
      if (frontCaptured && sideCaptured) {
        await stopStream(); // Stop stream on frontend
        await axios.post(`${API_BASE_URL}/shutdown`); // Stop stream on backend
      }
    } catch (error) {
      console.error('Error checking for images:', error);
    }
  };
  

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (streamState.isActive) {
      interval = setInterval(checkForImages, 1000);
    }
    return () => clearInterval(interval);
  }, [streamState.isActive]);

  const startStream = () => {
    try {
      setStreamState({
        isActive: true,
        stage: 'front',
        error: null,
        frontImage: null,
        sideImage: null,
        isComplete: false
      });
  
      if (imgRef.current) {
        imgRef.current.src = `${API_BASE_URL}/video_feed`;
      }
    } catch (error) {
      handleStreamError();
    }
  };
  

  const stopStream = async () => {
    setStreamState(prev => ({ ...prev, isActive: false }));
    
    if (imgRef.current) {
      imgRef.current.src = "";
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
      stopStream();
      await axios.post(`${API_BASE_URL}/restart`);
      setStreamState({
        isActive: false,
        stage: 'front',
        error: null,
        frontImage: null,
        sideImage: null,
        isComplete: false
      });
      setTimeout(startStream, 500);
    } catch (error) {
      setStreamState(prev => ({
        ...prev,
        error: 'Failed to restart process'
      }));
    }
  };

  return (
    <div className="w-full space-y-6">
      {/* Instructions */}
      <div className="bg-blue-50 p-4 rounded-lg">
        <h3 className="font-semibold text-blue-800 mb-2">Instructions:</h3>
        <p className="text-blue-600">
          {streamState.stage === 'front' 
            ? "Stand facing the camera with your arms slightly raised"
            : "Turn to your side and maintain the pose"}
        </p>
      </div>

      {/* Camera Feed or Captured Images */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Live Feed / Front Image */}
        <div className="relative bg-black w-96 h-96 rounded-xl overflow-hidden">
          {streamState.isActive ? (
            <img 
              ref={imgRef}
              src={`${API_BASE_URL}/video_feed`}
              alt="Camera Feed"
              className="w-full h-full object-contain"
              onError={handleStreamError}
            />
          ) : streamState.frontImage ? (
            <img 
              src={streamState.frontImage}
              alt="Front Pose"
              className="w-full h-full object-contain"
            />
          ) : (
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
          )}
        </div>

        {/* Side Image */}
        {(streamState.sideImage || streamState.stage === 'side') && (
          <div className="relative aspect-video bg-black rounded-xl overflow-hidden">
            {streamState.sideImage ? (
              <img 
                src={streamState.sideImage}
                alt="Side Pose"
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-white">
                Waiting for side pose...
              </div>
            )}
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex justify-between items-center">
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={restartProcess}
          className="flex items-center space-x-2 bg-gray-100 text-gray-700 px-6 py-3 rounded-lg hover:bg-gray-200"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Restart</span>
        </motion.button>

        {streamState.isActive && (
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={stopStream}
            className="bg-red-600 text-white px-6 py-3 rounded-lg hover:bg-red-700"
          >
            Stop Camera
          </motion.button>
        )}

        {streamState.isComplete && (
          <div className="text-green-600 font-semibold">
            Capture Complete! ✓
          </div>
        )}
      </div>

      {/* Error Display */}
      {streamState.error && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg"
        >
          {streamState.error}
        </motion.div>
      )}
    </div>
  );
}