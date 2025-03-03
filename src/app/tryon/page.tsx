'use client';
import { useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';
import { Suspense } from 'react';
import * as THREE from 'three';

export default function TryOnPage() {
  const [modelBlobUrl, setModelBlobUrl] = useState<string | null>(null);

  useEffect(() => {
    // Retrieve the Base64 stored model from Local Storage
    // const modelBase64 = localStorage.getItem("modelBlob");

    // if (modelBase64) {
    //   // Convert Base64 to Blob
    //   const byteCharacters = atob(modelBase64.split(',')[1]);
    //   const byteArrays = [];
    //   for (let i = 0; i < byteCharacters.length; i++) {
    //     byteArrays.push(byteCharacters.charCodeAt(i));
    //   }
    //   const byteArray = new Uint8Array(byteArrays);
    //   const blob = new Blob([byteArray], { type: "application/octet-stream" });

    //   // Create an Object URL for the Blob
    //   const objectUrl = URL.createObjectURL(blob);
    //   setModelBlobUrl(objectUrl);

    setModelBlobUrl('/models/test.obj');
  

    }
  , []);

  return (
    <div className="w-screen h-screen bg-gray-100 flex flex-col items-center justify-center">
      <h1 className="text-3xl font-bold mb-4">3D Model Try-On</h1>
      <div className="w-full h-[600px] bg-white rounded-lg shadow-md">
        {modelBlobUrl ? (
          <Canvas camera={{ position: [0, 2, 5] }} shadows>
            <ambientLight intensity={0.5} />
            <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} />
            <pointLight position={[-10, -10, -10]} />
            <Suspense fallback={<p>Loading...</p>}>
              <Model url={modelBlobUrl} />
            </Suspense>
            <OrbitControls enableZoom={true} />
          </Canvas>
        ) : (
          <p>Loading model...</p>
        )}
      </div>
    </div>
  );
}

function Model({ url }: { url: string }) {
  const { scene } = useGLTF(url);

  scene.traverse((child) => {
    if ((child as THREE.Mesh).isMesh) {
      const mesh = child as THREE.Mesh;
      mesh.material = new THREE.MeshStandardMaterial({
        color: 'gray',
        metalness: 0.5,
        roughness: 0.7
      });
    }
  });

  return <primitive object={scene} scale={1.5} />;
}
