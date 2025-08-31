import React from 'react';
import { Link } from 'react-router-dom';
import { Play, Shield, Zap, Target, ChevronRight, Eye } from 'lucide-react';

function HomePage() {
  return (
    <div className="relative overflow-hidden">
      {/* Hero Section */}
      <section className="relative px-4 sm:px-6 lg:px-8 py-20 lg:py-32">
        <div className="max-w-7xl mx-auto">
          <div className="text-center">
            <div className="flex justify-center mb-8">
              <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center shadow-2xl">
                <Eye className="w-10 h-10 text-white" />
              </div>
            </div>
            
            <h1 className="text-4xl sm:text-6xl lg:text-7xl font-bold text-white mb-6 leading-tight">
              Smart Lie{' '}
              <span className="bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                Detector
              </span>
            </h1>
            <p className="text-xl sm:text-2xl text-slate-300 mb-12 max-w-3xl mx-auto leading-relaxed">
              Harness the power of advanced AI to detect deception through vocal patterns, 
              micro-expressions, and behavioral analysis.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Link
                to="/capture"
                className="group inline-flex items-center px-8 py-4 bg-gradient-to-r from-blue-600 to-blue-700 text-white font-semibold rounded-xl shadow-lg hover:shadow-blue-500/25 hover:shadow-2xl transform hover:scale-105 transition-all duration-300"
              >
                <Play className="w-5 h-5 mr-2 group-hover:scale-110 transition-transform duration-200" />
                Start Analysis
                <ChevronRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform duration-200" />
              </Link>
              
              <button className="inline-flex items-center px-8 py-4 border-2 border-slate-600 text-slate-300 font-semibold rounded-xl hover:border-slate-500 hover:text-white hover:bg-slate-800/50 transition-all duration-300">
                Learn More
              </button>
            </div>
          </div>
        </div>

        {/* Background Elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-500/10 rounded-full blur-3xl"></div>
          <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-purple-500/10 rounded-full blur-3xl"></div>
        </div>
      </section>

      {/* Features Section */}
      <section className="px-4 sm:px-6 lg:px-8 py-20">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
              Advanced Detection Technology
            </h2>
            <p className="text-lg text-slate-400 max-w-2xl mx-auto">
              Our cutting-edge AI analyzes multiple behavioral indicators to provide accurate deception detection.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-8 hover:bg-slate-800/70 transition-all duration-300 group">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">Real-time Analysis</h3>
              <p className="text-slate-400 leading-relaxed">
                Instant processing of vocal patterns, facial expressions, and behavioral cues during live recording sessions.
              </p>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-8 hover:bg-slate-800/70 transition-all duration-300 group">
              <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                <Target className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">High Accuracy</h3>
              <p className="text-slate-400 leading-relaxed">
                Advanced machine learning algorithms trained on extensive datasets deliver industry-leading accuracy rates.
              </p>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-8 hover:bg-slate-800/70 transition-all duration-300 group">
              <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-green-600 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                <Shield className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">Secure & Private</h3>
              <p className="text-slate-400 leading-relaxed">
                Your recordings are processed locally with enterprise-grade security. No data leaves your device.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="px-4 sm:px-6 lg:px-8 py-20">
        <div className="max-w-4xl mx-auto text-center">
          <div className="bg-gradient-to-r from-slate-800/80 to-slate-700/80 backdrop-blur-sm border border-slate-600/50 rounded-3xl p-12">
            <h2 className="text-3xl sm:text-4xl font-bold text-white mb-6">
              Ready to Detect Deception?
            </h2>
            <p className="text-lg text-slate-300 mb-8">
              Experience the future of truth detection with our advanced AI-powered analysis system.
            </p>
            <Link
              to="/capture"
              className="inline-flex items-center px-10 py-5 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl shadow-lg hover:shadow-blue-500/25 hover:shadow-2xl transform hover:scale-105 transition-all duration-300 group"
            >
              <Play className="w-6 h-6 mr-3 group-hover:scale-110 transition-transform duration-200" />
              Begin Analysis
              <ChevronRight className="w-6 h-6 ml-3 group-hover:translate-x-1 transition-transform duration-200" />
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}

export default HomePage;