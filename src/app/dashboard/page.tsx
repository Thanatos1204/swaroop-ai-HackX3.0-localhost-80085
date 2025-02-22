// src/app/page.tsx
'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { WavyBackground } from "@/components/ui/wavy-background";
import { useState } from 'react';
import { Camera, Upload, Ruler, ShirtIcon, CheckCircle, Store, History, ArrowRight, Sparkles } from 'lucide-react';
import Link from 'next/link';
import { Timeline } from "@/components/ui/timeline";
// Interfaces remain the same
interface Feature {
    title: string;
    description: string;
    icon: React.ReactNode;
    buttonText: string;
    buttonIcon?: React.ReactNode;
    gradient: string;
}

interface BrandLocation {
    name: string;
    address: string;
    distance: string;
}

interface PastOrder {
    id: string;
    date: string;
    items: string[];
    status: string;
}

// Animation variants remain the same
const fadeInUp = {
    hidden: { opacity: 0, y: 20 },
    visible: {
        opacity: 1,
        y: 0,
        transition: { duration: 0.6 }
    }
};

const staggerContainer = {
    hidden: { opacity: 0 },
    visible: {
        opacity: 1,
        transition: {
            staggerChildren: 0.2
        }
    }
};

const pulseAnimation = {
    scale: [1, 1.02, 1],
    transition: {
        duration: 1.5,
        repeat: Infinity,
        ease: "easeInOut"
    }
};

export default function LandingPage() {
    // State declarations remain the same
    const [height, setHeight] = useState('');
    const [files, setFiles] = useState<{ front?: File; side?: File }>({});
    const [hoveredCard, setHoveredCard] = useState<number | null>(null);
    const timelineData = [
        {
          title: "Smart Size Detection",
          content: (
            <div className="bg-white dark:bg-neutral-900 rounded-lg p-6 shadow-lg">
              <h4 className="text-lg font-semibold mb-3">AI-Powered Size Analysis</h4>
              <p className="text-gray-600 dark:text-gray-300">
                Implementing cutting-edge machine learning algorithms to analyze photos and measurements for accurate size recommendations. Achieved 95% accuracy in size predictions.
              </p>
            </div>
          ),
        },
        {
          title: "Virtual Try-On",
          content: (
            <div className="bg-white dark:bg-neutral-900 rounded-lg p-6 shadow-lg">
              <h4 className="text-lg font-semibold mb-3">3D Modeling Integration</h4>
              <p className="text-gray-600 dark:text-gray-300">
                Developed a real-time virtual fitting room using advanced 3D modeling technology. Users can now visualize how clothes will look on them before purchase.
              </p>
            </div>
          ),
        },
        {
          title: "Brand Matching",
          content: (
            <div className="bg-white dark:bg-neutral-900 rounded-lg p-6 shadow-lg">
              <h4 className="text-lg font-semibold mb-3">Smart Brand Recommendations</h4>
              <p className="text-gray-600 dark:text-gray-300">
                Created an intelligent brand matching system that connects users with their perfect size across different brands, considering size variations between manufacturers.
              </p>
            </div>
          ),
        },
      ];
    // Features array remains the same
    const features: Feature[] = [
        {
            title: "Smart Size Detection",
            description: "Upload your photos or enter your measurements for perfect size recommendations",
            icon: <Ruler className="w-6 h-6 text-white" />,
            buttonText: "Get Your Size",
            buttonIcon: <Ruler className="w-4 h-4 mr-2" />,
            gradient: "from-blue-400 to-blue-600"
        },
        {
            title: "Virtual Try-On",
            description: "See how clothes will look on you before making a purchase",
            icon: <ShirtIcon className="w-6 h-6 text-white" />,
            buttonText: "Try Now",
            buttonIcon: <Camera className="w-4 h-4 mr-2" />,
            gradient: "from-purple-400 to-purple-600"
        },
        {
            title: "Brand Matching",
            description: "Get recommendations from your favorite brands in your perfect size",
            icon: <Store className="w-6 h-6 text-white" />,
            buttonText: "Find Brands",
            buttonIcon: <Store className="w-4 h-4 mr-2" />,
            gradient: "from-indigo-400 to-indigo-600"
        }
    ];

    const pastOrders: PastOrder[] = [
        {
            id: "ORD001",
            date: "2024-02-20",
            items: ["Blue Denim Jacket", "White T-Shirt"],
            status: "Delivered"
        },
        {
            id: "ORD002",
            date: "2024-02-15",
            items: ["Black Jeans", "Sneakers"],
            status: "Delivered"
        }
    ];

    const nearbyBrands: BrandLocation[] = [
        {
            name: "Fashion Hub",
            address: "123 Fashion Street",
            distance: "0.5 km"
        },
        {
            name: "Style Center",
            address: "456 Style Avenue",
            distance: "1.2 km"
        },
        {
            name: "Trend Store",
            address: "789 Trend Boulevard",
            distance: "2.0 km"
        }
    ];

    return (
        <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white flex flex-col">
            {/* Hero Section */}
            <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
                <WavyBackground
                    className="absolute inset-0 z-0"
                    colors={["#38bdf8", "#818cf8", "#c084fc"]}
                    waveWidth={60}
                    backgroundFill="#f0f9ff"
                    blur={15}
                    speed="fast"
                    waveOpacity={0.4}
                />

                <motion.div
                    initial="hidden"
                    animate="visible"
                    variants={fadeInUp}
                    className="relative z-10 w-full max-w-6xl mx-auto px-4 flex flex-col items-center"
                >
                    <motion.div
                        variants={fadeInUp}
                        className="text-center mb-12"
                    >
                        <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6">
                            Find Your Perfect Fit
                        </h1>
                        <p className="text-xl text-gray-600 max-w-2xl mx-auto">
                            Get personalized size recommendations using AI technology.
                        </p>
                    </motion.div>

                    <motion.div
                        variants={fadeInUp}
                        className="max-w-4xl w-full mx-auto bg-white rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-shadow duration-300"
                    >
                        {/* Form content remains the same */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 align-middle">
                            {/* Left side with height input and photo upload remains the same */}
                            <motion.div
                                className="space-y-6"
                                variants={staggerContainer}
                            >
                                {/* Height input section remains the same */}
                                <motion.div variants={fadeInUp}>
                                    <label className="block text-sm font-medium text-gray-700 mb-2">
                                        Height (cm)
                                    </label>
                                    <input
                                        type="number"
                                        value={height}
                                        onChange={(e) => setHeight(e.target.value)}
                                        className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300"
                                        placeholder="Enter your height"
                                    />
                                </motion.div>

                                {/* Photo upload section remains the same */}
                                <motion.div variants={fadeInUp} className="space-y-4">
                                    <label className="block text-sm font-medium text-gray-700">
                                        Upload Photos
                                    </label>
                                    <div className="grid grid-cols-2 gap-4">
                                        {['front', 'side'].map((view) => (
                                            <motion.div
                                                key={view}
                                                whileHover={{ scale: 1.02 }}
                                                whileTap={{ scale: 0.98 }}
                                                className="relative"
                                            >
                                                <div className="h-40 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center bg-gray-50 hover:bg-gray-100 transition-all duration-300 hover:border-blue-500">
                                                    <div className="text-center">
                                                        <motion.div
                                                            animate={{ rotate: [0, 0] }}
                                                            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                                                        >
                                                            <Camera className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                                                        </motion.div>
                                                        <span className="text-sm text-gray-500">{view === 'front' ? 'Front View' : 'Side View'}</span>
                                                    </div>
                                                    <input
                                                        type="file"
                                                        className="absolute inset-0 opacity-0 cursor-pointer"
                                                        onChange={(e) => {
                                                            if (e.target.files?.[0]) {
                                                                setFiles(prev => ({ ...prev, [view]: e.target.files![0] }));
                                                            }
                                                        }}
                                                        accept="image/*"
                                                    />
                                                </div>
                                            </motion.div>
                                        ))}
                                    </div>
                                </motion.div>
                            </motion.div>

                            {/* Size recommendation panel remains the same */}
                            <motion.div
                                variants={fadeInUp}
                                className="bg-gray-50 rounded-lg p-6"
                                whileHover={{ scale: 1.01 }}
                                transition={{ duration: 0.3 }}
                            >
                                <h3 className="text-lg font-semibold mb-4">Your Perfect Size</h3>
                                <div className="space-y-4">
                                    <motion.div
                                        className="flex items-center justify-between"
                                        animate={pulseAnimation}
                                    >
                                        <span className="text-gray-600">Size:</span>
                                        <span className="font-medium">Loading...</span>
                                    </motion.div>
                                    <motion.div
                                        className="flex items-center justify-between"
                                        animate={pulseAnimation}
                                    >
                                        <span className="text-gray-600">Fit:</span>
                                        <span className="font-medium">Loading...</span>
                                    </motion.div>
                                    <motion.button
                                        whileHover={{ scale: 1.02 }}
                                        whileTap={{ scale: 0.98 }}
                                        className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 transition-all duration-300"
                                    >
                                        Get Recommendation
                                    </motion.button>
                                </div>
                            </motion.div>
                        </div>
                    </motion.div>
                </motion.div>
            </section>

            {/* Features Section remains the same */}
            <motion.section
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                variants={staggerContainer}
                className="py-16 bg-white"
            >
                <div className="container mx-auto px-8">
                    <motion.h2
                        variants={fadeInUp}
                        className="text-5xl font-bold text-center mb-12"
                    >
                        How It Works
                    </motion.h2>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                        {features.map((feature, index) => (
                            <motion.div
                                key={index}
                                variants={fadeInUp}
                                whileHover={{ y: -5 }}
                                onHoverStart={() => setHoveredCard(index)}
                                onHoverEnd={() => setHoveredCard(null)}
                                className="relative group"
                            >
                                <div className="bg-white rounded-xl p-12 shadow-lg hover:shadow-xl transition-all duration-300">
                                    <motion.div
                                        animate={{
                                            rotate: hoveredCard === index ? [0, 360] : 0
                                        }}
                                        transition={{ duration: 1 }}
                                        className={`w-12 h-12 rounded-lg flex items-center justify-center mb-4 bg-gradient-to-r ${feature.gradient}`}
                                    >
                                        {feature.icon}
                                    </motion.div>
                                    <h3 className="text-xl font-semibold mb-6">{feature.title}</h3>
                                    <p className="text-gray-600 mb-12">{feature.description}</p>
                                    <motion.button
                                        whileHover={{ scale: 1.02 }}
                                        whileTap={{ scale: 0.98 }}
                                        className="w-full flex items-center justify-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all duration-300 group"
                                    >
                                        <span className="flex items-center">
                                            {feature.buttonIcon}
                                            {feature.buttonText}
                                            <motion.div
                                                animate={{ x: [0, 4, 0] }}
                                                transition={{ duration: 1, repeat: Infinity }}
                                            >
                                                <ArrowRight className="ml-2 w-4 h-4" />
                                            </motion.div>
                                        </span>
                                    </motion.button>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </motion.section>


            <motion.section
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                variants={staggerContainer}
                className="py-20 bg-gray-50"
            >
                <div className="container mx-auto px-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
                        {/* Past Orders */}
                        <motion.div variants={fadeInUp}>
                            <h2 className="text-3xl font-bold mb-8 flex items-center">
                                <History className="w-8 h-8 mr-3 text-blue-600" />
                                Previous Orders
                            </h2>
                            <motion.div
                                variants={staggerContainer}
                                className="space-y-6"
                            >
                                {pastOrders.map((order) => (
                                    <motion.div
                                        key={order.id}
                                        variants={fadeInUp}
                                        whileHover={{ scale: 1.02, y: -5 }}
                                        className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-100"
                                    >
                                        <div className="flex justify-between items-start mb-4">
                                            <div>
                                                <h3 className="text-xl font-semibold text-gray-800">
                                                    Order #{order.id}
                                                </h3>
                                                <p className="text-gray-500 mt-1">{order.date}</p>
                                            </div>
                                            <motion.span
                                                whileHover={{ scale: 1.1 }}
                                                className="px-4 py-1.5 bg-green-100 text-green-700 rounded-full text-sm font-medium"
                                            >
                                                {order.status}
                                            </motion.span>
                                        </div>
                                        <motion.div
                                            initial={{ opacity: 0.8 }}
                                            whileHover={{ opacity: 1 }}
                                            className="mt-4 p-4 bg-gray-50 rounded-lg"
                                        >
                                            <div className="flex flex-wrap gap-2">
                                                {order.items.map((item, idx) => (
                                                    <span
                                                        key={idx}
                                                        className="px-3 py-1 bg-white rounded-full text-sm text-gray-700 shadow-sm"
                                                    >
                                                        {item}
                                                    </span>
                                                ))}
                                            </div>
                                        </motion.div>
                                    </motion.div>
                                ))}
                            </motion.div>
                        </motion.div>

                        {/* Available Branches */}
                        <motion.div variants={fadeInUp}>
                            <h2 className="text-3xl font-bold mb-8 flex items-center">
                                <Store className="w-8 h-8 mr-3 text-blue-600" />
                                Available Branches
                            </h2>
                            <motion.div
                                variants={staggerContainer}
                                className="space-y-6"
                            >
                                {nearbyBrands.map((brand, index) => (
                                    <motion.div
                                        key={index}
                                        variants={fadeInUp}
                                        whileHover={{ scale: 1.02, y: -5 }}
                                        className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-100"
                                    >
                                        <div className="flex items-start justify-between">
                                            <div className="flex-grow">
                                                <h3 className="text-xl font-semibold text-gray-800 mb-2">
                                                    {brand.name}
                                                </h3>
                                                <p className="text-gray-600">{brand.address}</p>
                                            </div>
                                            <motion.div
                                                whileHover={{ scale: 1.1 }}
                                                className="flex items-center px-4 py-1.5 bg-blue-50 text-blue-600 rounded-full"
                                            >
                                                <span className="text-sm font-medium">{brand.distance}</span>
                                            </motion.div>
                                        </div>
                                        <motion.button
                                            whileHover={{ scale: 1.02 }}
                                            whileTap={{ scale: 0.98 }}
                                            className="mt-4 w-full flex items-center justify-center px-4 py-2 bg-gray-50 text-gray-700 rounded-lg hover:bg-gray-100 transition-colors duration-300"
                                        >
                                            <span className="flex items-center">
                                                View Details
                                                <ArrowRight className="ml-2 w-4 h-4" />
                                            </span>
                                        </motion.button>
                                    </motion.div>
                                ))}
                            </motion.div>
                        </motion.div>
                    </div>
                </div>
            </motion.section>
            {/* Timeline Section */}
            <motion.section
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                variants={staggerContainer}
                className="py-16 bg-white"
            >
                <Timeline data={timelineData} />
            </motion.section>
        </div>
    );
}