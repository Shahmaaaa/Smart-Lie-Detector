import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Eye, Home, Video, BarChart3 } from 'lucide-react';

interface LayoutProps {
  children: React.ReactNode;
}

function Layout({ children }: LayoutProps) {
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="bg-slate-900/80 backdrop-blur-sm border-b border-slate-700/50 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <Link to="/" className="flex items-center space-x-3 group">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center group-hover:scale-105 transition-transform duration-200">
                <Eye className="w-6 h-6 text-white" />
              </div>
              <span className="text-xl font-bold text-white group-hover:text-blue-400 transition-colors duration-200">
                Smart Lie Detector
              </span>
            </Link>

            <nav className="flex space-x-1">
              <Link
                to="/"
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                  isActive('/') 
                    ? 'bg-blue-600 text-white shadow-lg' 
                    : 'text-slate-300 hover:text-white hover:bg-slate-700/50'
                }`}
              >
                <Home className="w-4 h-4" />
                <span className="hidden sm:inline">Home</span>
              </Link>
              <Link
                to="/capture"
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                  isActive('/capture') 
                    ? 'bg-blue-600 text-white shadow-lg' 
                    : 'text-slate-300 hover:text-white hover:bg-slate-700/50'
                }`}
              >
                <Video className="w-4 h-4" />
                <span className="hidden sm:inline">Capture</span>
              </Link>
              <Link
                to="/results"
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                  isActive('/results') 
                    ? 'bg-blue-600 text-white shadow-lg' 
                    : 'text-slate-300 hover:text-white hover:bg-slate-700/50'
                }`}
              >
                <BarChart3 className="w-4 h-4" />
                <span className="hidden sm:inline">Results</span>
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative">
        {children}
      </main>

      {/* Footer */}
      <footer className="bg-slate-900/50 border-t border-slate-700/50 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-slate-400">
            <p>&copy; 2025 Smart Lie Detector. Advanced AI-powered deception detection.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default Layout;