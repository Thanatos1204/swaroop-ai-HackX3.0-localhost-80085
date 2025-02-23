export interface User {
    id: string;
    name: string;
    email: string;
  }
  
  export interface Order {
    id: string;
    date: string;
    items: ClothingItem[];
    totalAmount: number;
  }
  
  export interface ClothingItem {
    id: string;
    name: string;
    brand: string;
    size: string;
    fit: string;
    price: number;
    imageUrl: string;
  }
  
  export interface SizeRecommendation {
    size: string;
    fit: string;
    confidence: number;
  }