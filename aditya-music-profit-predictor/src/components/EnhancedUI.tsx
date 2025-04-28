import { useState, useEffect, ReactNode } from 'react';

interface ThemeConfig {
  mainColor: string;
  accentColor: string;
  bgGradientFrom: string;
  bgGradientTo: string;
  textColor: string;
  cardBg: string;
  cardHover: string;
  glassEffect: string;
  accentGradient: string;
  secondaryGradient: string;
  highlightColor: string;
  buttonGradient: string;
}

export const themes: Record<string, ThemeConfig> = {
  dark: {
    mainColor: 'from-gray-900 to-slate-800',
    accentColor: 'bg-blue-600 hover:bg-blue-700',
    bgGradientFrom: '#1a202c',
    bgGradientTo: '#111827',
    textColor: 'text-gray-50',
    cardBg: 'bg-slate-800/90',
    cardHover: 'hover:bg-slate-700/80',
    glassEffect: 'backdrop-blur-md bg-slate-800/40 border border-slate-700/50',
    accentGradient: 'bg-gradient-to-r from-blue-600 to-indigo-600',
    secondaryGradient: 'bg-gradient-to-r from-purple-600 to-pink-600',
    highlightColor: 'text-blue-400',
    buttonGradient: 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700'
  },
  blue: {
    mainColor: 'from-blue-900 to-indigo-900',
    accentColor: 'bg-cyan-500 hover:bg-cyan-600',
    bgGradientFrom: '#1e3a8a',
    bgGradientTo: '#312e81',
    textColor: 'text-white',
    cardBg: 'bg-blue-800/80',
  },
  purple: {
    mainColor: 'from-purple-900 to-indigo-900',
    accentColor: 'bg-pink-600 hover:bg-pink-700',
    bgGradientFrom: '#581c87',
    bgGradientTo: '#312e81',
    textColor: 'text-white',
    cardBg: 'bg-purple-800/80',
  },
  neon: {
    mainColor: 'from-gray-900 to-black',
    accentColor: 'bg-green-500 hover:bg-green-600',
    bgGradientFrom: '#0f0f0f',
    bgGradientTo: '#000000',
    textColor: 'text-white',
    cardBg: 'bg-gray-900/90',
  },
  luxury: {
    mainColor: 'from-gray-900 to-gray-800',
    accentColor: 'bg-amber-600 hover:bg-amber-700',
    bgGradientFrom: '#1a1a1a',
    bgGradientTo: '#262626',
    textColor: 'text-amber-50',
    cardBg: 'bg-gray-800/90',
  }
};

export const useCurrentTheme = (themeType = 'dark') => {
  const themes = {
    dark: {
      mainColor: 'from-gray-900 to-slate-800',
      cardBg: 'bg-slate-800/90',
      cardHover: 'hover:bg-slate-700/80',
      textColor: 'text-gray-50',
      glassEffect: 'backdrop-blur-md bg-slate-800/40 border border-slate-700/50',
      accentGradient: 'bg-gradient-to-r from-blue-600 to-indigo-600',
      secondaryGradient: 'bg-gradient-to-r from-purple-600 to-pink-600',
      highlightColor: 'text-blue-400',
      buttonGradient: 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700'
    },
    // ... other themes
  };

  return { themeConfig: themes[themeType || 'dark'] };
};

interface AnimatedCardProps {
  children: ReactNode;
  delay?: number;
  className?: string;
}

export const AnimatedCard = ({ children, className = '', delay = 0 }) => {
  return (
    <div
      className={`transition-all duration-700 ease-out transform translate-y-0 opacity-100 ${className}`}
      style={{ 
        animationDelay: `${delay}ms`,
        boxShadow: '0 10px 30px rgba(0, 0, 0, 0.15)'
      }}
    >
      {children}
    </div>
  );
};

interface GlassCardProps {
  children: ReactNode;
  className?: string;
}

export const GlassCard = ({ children, className = '' }) => {
  const { themeConfig } = useCurrentTheme();
  
  return (
    <div 
      className={`rounded-xl ${themeConfig.glassEffect} p-6 ${className}`}
      style={{
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)',
      }}
    >
      {children}
    </div>
  );
};

interface PulsatingIconProps {
  icon: string;
  size?: string;
  color?: string;
}

export function PulsatingIcon({ icon, size = 'text-2xl', color = 'text-blue-500' }: PulsatingIconProps) {
  return (
    <span className={`inline-block ${size} ${color} animate-pulse`}>
      {icon}
    </span>
  );
}

interface ShimmerButtonProps {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  className?: string;
}

export const ShimmerButton = ({ children, onClick, disabled = false, className = '' }) => {
  const { themeConfig } = useCurrentTheme();
  
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`relative px-6 py-3 rounded-lg font-medium text-white overflow-hidden transition-all duration-300 
        ${disabled ? 'bg-gray-600 cursor-not-allowed' : `${themeConfig.buttonGradient} transform hover:scale-105`} 
        ${className}`}
      style={{
        boxShadow: disabled ? 'none' : '0 4px 15px rgba(59, 130, 246, 0.3)'
      }}
    >
      <span className="relative z-10">{children}</span>
      {!disabled && (
        <div className="absolute inset-0 w-full h-full opacity-30">
          <div className="absolute inset-0 w-1/3 h-full bg-white transform -skew-x-12 translate-x-[-150%] shimmer"></div>
        </div>
      )}
      <style jsx>{`
        .shimmer {
          animation: shimmer 3s infinite;
        }
        @keyframes shimmer {
          0% { transform: translateX(-150%) skewX(-12deg); }
          50% { transform: translateX(150%) skewX(-12deg); }
          100% { transform: translateX(150%) skewX(-12deg); }
        }
      `}</style>
    </button>
  );
};

export function ResponsiveGrid({ children }: { children: ReactNode }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
      {children}
    </div>
  );
}

export function ThemeSelector() {
  const { currentTheme, toggleTheme } = useCurrentTheme();
  
  const themeButtons = Object.keys(themes).map(themeName => {
    const colors = themes[themeName];
    const isActive = currentTheme === themeName;
    
    return (
      <button
        key={themeName}
        onClick={() => toggleTheme(themeName)}
        className={`w-8 h-8 rounded-full border-2 ${isActive ? 'border-white scale-125' : 'border-transparent'} overflow-hidden transition-transform duration-200`}
        style={{ 
          background: `linear-gradient(to bottom right, ${colors.bgGradientFrom}, ${colors.bgGradientTo})` 
        }}
        title={`${themeName.charAt(0).toUpperCase()}${themeName.slice(1)} Theme`}
      />
    );
  });
  
  return (
    <div className="flex items-center space-x-2">
      <span className="text-sm mr-1">Theme:</span>
      {themeButtons}
    </div>
  );
}

export function AddGlobalStyles() {
  useEffect(() => {
    // Add the shimmer animation to the global styles
    const style = document.createElement('style');
    style.innerHTML = `
      @keyframes shimmer {
        0% {
          transform: translateX(-100%);
        }
        100% {
          transform: translateX(100%);
        }
      }
      
      .shimmer {
        animation: shimmer 2s infinite;
      }
      
      :root {
        --bg-gradient-from: #1a202c;
        --bg-gradient-to: #111827;
      }
      
      body {
        background: linear-gradient(to bottom, var(--bg-gradient-from), var(--bg-gradient-to));
        min-height: 100vh;
        background-attachment: fixed;
      }
    `;
    
    document.head.appendChild(style);
    
    return () => {
      document.head.removeChild(style);
    };
  }, []);
  
  return null;
}

// Export all components as a bundle
export default function EnhancedUI() {
  return <AddGlobalStyles />;
} 