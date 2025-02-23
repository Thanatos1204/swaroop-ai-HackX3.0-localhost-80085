// src/app/page.tsx
'use client';
import { motion } from 'framer-motion';
import { WavyBackground } from "@/components/ui/wavy-background";
import { useState } from 'react';
import { Camera, Ruler, ShirtIcon, Store, History, ArrowRight } from 'lucide-react';
import { Timeline } from "@/components/ui/timeline";
import axios from 'axios';

// Form Components
interface FormInputProps {
  label: string;
  type: string;
  value: string | number;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  placeholder: string;
}

const FormInput = ({ label, type, value, onChange, placeholder }: FormInputProps) => (
  <motion.div variants={fadeInUp} className="mb-6">
    <label className="block text-sm font-medium text-gray-700 mb-2">
      {label}
    </label>
    <input
      type={type}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300"
    />
  </motion.div>
);

const GenderSelect = ({ value, onChange }: { value: string; onChange: (e: React.ChangeEvent<HTMLSelectElement>) => void }) => (
  <motion.div variants={fadeInUp} className="mb-6">
    <label className="block text-sm font-medium text-gray-700 mb-2">
      Gender
    </label>
    <select
      value={value}
      onChange={onChange}
      className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300"
    >
      <option value="">Select Gender</option>
      <option value="male">Male</option>
      <option value="female">Female</option>
      <option value="other">Other</option>
    </select>
  </motion.div>
);

// Form Navigation
interface FormNavbarProps {
  activeForm: string;
  setActiveForm: (form: string) => void;
}

const FormNavbar = ({ activeForm, setActiveForm }: FormNavbarProps) => {
  const formOptions = [
    { id: 'photo-height', label: 'Photo & Height' },
    { id: 'height-only', label: 'Height Only' },
    { id: 'height-weight', label: 'Height & Weight' },
    { id: 'live-cam', label: 'Live Camera' }
  ];

  return (
    <nav className="sticky top-0 z-50 bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          <div className="hidden sm:flex sm:space-x-4 w-full justify-center">
            {formOptions.map((option) => (
              <motion.button
                key={option.id}
                onClick={() => setActiveForm(option.id)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className={`px-6 py-2 rounded-full text-sm font-medium transition-all duration-300 ${activeForm === option.id
                    ? 'bg-blue-600 text-white shadow-md'
                    : 'bg-gray-50 text-gray-700 hover:bg-gray-100'
                  }`}
              >
                {option.label}
              </motion.button>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};

// Animation variants
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

// Interfaces
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

export default function LandingPage() {
  // States
  // Add these states at the top of your component
  const [isLoading, setIsLoading] = useState(false);
  const [sizeRecommendation, setSizeRecommendation] = useState<{ recommended_size: string; } | null>(null);
  const [activeForm, setActiveForm] = useState('photo-height');
  const [height, setHeight] = useState(0);
  const [weight, setWeight] = useState(0);
  const [gender, setGender] = useState('');
  const [age, setAge] = useState('');
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
  const handleAPICall = async (event: React.MouseEvent) => {
    event.preventDefault();
    setIsLoading(true);
    try {
      switch (activeForm) {

        case 'height-only':
          // Height & Gender Only Form
          if (!height || !gender) {
            alert('Please provide height and gender');
            return;
          }
          const heightResponse = await axios.post(
            `${process.env.NEXT_PUBLIC_API_URL}/get-size-height-gender`,
            {
              height: Number(height),
              gender: gender
            }
          );
          setSizeRecommendation(heightResponse.data);
          console.log(heightResponse.data);
          break;

        case 'height-weight':
          // Height, Weight, Gender & Age Form
          if (!height || !weight || !gender || !age) {
            alert('Please fill all fields');
            return;
          }
          const fullResponse = await axios.post(
            `${process.env.NEXT_PUBLIC_API_URL}/get-size`,
            {
              height: Number(height),
              weight: Number(weight),
              gender: gender,
              age: Number(age)
            }
          );
          setSizeRecommendation(fullResponse.data);
          console.log(fullResponse.data);
          break;

        case 'live-cam':
          // Live Camera Form
          // if (!files.webcam) {
          //   setError('Please capture an image first');
          //   return;
          // }
          // const webcamData = new FormData();
          // webcamData.append('image', files.webcam);

          const webcamResponse = await axios.post(
            'https://your-ngrok-url/webcam-prediction',
            // webcamData,
            {
              headers: { 'Content-Type': 'multipart/form-data' }
            }
          );
          setSizeRecommendation(webcamResponse.data);
          break;
      }
    } catch (error) {
      console.error('Error:', error);
    } finally {
      alert('Failed to get size recommendation. Please try again.');
    }
  };
  // Form renderer based on active form
  const renderForm = () => {
    switch (activeForm) {


      case 'height-only':
        return (
          <>
            <FormInput
              label="Height (cm)"
              type="number"
              value={height}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setHeight(Number(e.target.value))}
              placeholder="Enter your height"
            />
            <GenderSelect
              value={gender}
              onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setGender(e.target.value)}
            />
          </>
        );

      case 'height-weight':
        return (
          <>
            <FormInput
              label="Height (cm)"
              type="number"
              value={height}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setHeight(Number(e.target.value))}
              placeholder="Enter your height"
            />
            <FormInput
              label="Weight (kg)"
              type="number"
              value={weight}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setWeight(Number(e.target.value))}
              placeholder="Enter your weight"
            />
            <GenderSelect
              value={gender}
              onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setGender(e.target.value)}
            />
            <FormInput
              label="Age"
              type="number"
              value={age}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setAge(e.target.value)}
              placeholder="Enter your age"
            />
          </>
        );

      case 'live-cam':
        return (
          <div className="h-64 bg-gray-100 rounded-lg flex flex-col items-center justify-center">
            <Camera className="w-16 h-16 text-gray-400 mb-4" />
            <p className="text-gray-500">Camera Preview</p>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="mt-4 px-6 py-2 bg-blue-600 text-white rounded-lg"
            >
              Start Camera
            </motion.button>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white flex flex-col">
      <FormNavbar activeForm={activeForm} setActiveForm={setActiveForm} />

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
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 align-middle">
              <motion.div className="space-y-6" variants={staggerContainer}>
                {renderForm()}
              </motion.div>

              {/* Recommendation Panel */}
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
                    <span className="font-medium">{isLoading ? 'Loading...' : sizeRecommendation?.recommended_size}</span>
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
                    onClick={handleAPICall}
                    disabled={isLoading}
                    className={`w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 
    transition-all duration-300 ${isLoading ? 'opacity-75 cursor-not-allowed' : ''}`}
                  >
                    {isLoading ? (
                      <span className="flex items-center justify-center">
                        <motion.div
                          animate={{ rotate: 360 }}
                          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                          className="w-5 h-5 border-2 border-white border-t-transparent rounded-full mr-2"
                        />
                        Processing...
                      </span>
                    ) : (
                      'Get Recommendation'
                    )}
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