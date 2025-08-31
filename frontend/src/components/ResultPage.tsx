import { useLocation, Link } from 'react-router-dom';
import { 
  CheckCircle, 
  AlertTriangle, 
  BarChart3, 
  Eye, 
  Mic, 
  User,
  ArrowLeft,
  RotateCcw
} from 'lucide-react';

function ResultPage() {
  const location = useLocation();
  const result = location.state?.result;

  if (!result) {
    return (
      <div className="px-4 sm:px-6 lg:px-8 py-20">
        <div className="max-w-2xl mx-auto text-center">
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-3xl p-12">
            <AlertTriangle className="w-16 h-16 text-yellow-500 mx-auto mb-6" />
            <h1 className="text-3xl font-bold text-white mb-4">No Results Found</h1>
            <p className="text-lg text-slate-400 mb-8">
              No analysis data available. Please start a new recording session.
            </p>
            <Link
              to="/"
              className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-700 text-white font-semibold rounded-xl shadow-lg hover:shadow-blue-500/25 hover:shadow-xl transform hover:scale-105 transition-all duration-300"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Go Home
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const isDeceptionDetected = result.decision.includes('Deception');
  const confidencePercentage = (result.confidence * 100).toFixed(1);

  return (
    <div className="px-4 sm:px-6 lg:px-8 py-12">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-3xl sm:text-4xl font-bold text-white mb-4">
            Analysis Results
          </h1>
          <p className="text-lg text-slate-400">
            Advanced AI analysis complete
          </p>
        </div>

        {/* Main Result Card */}
        <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-3xl p-8 mb-8">
          <div className="text-center mb-8">
            <div className={`inline-flex items-center justify-center w-20 h-20 rounded-full mb-6 ${
              isDeceptionDetected 
                ? 'bg-gradient-to-br from-red-500 to-red-600' 
                : 'bg-gradient-to-br from-green-500 to-green-600'
            }`}>
              {isDeceptionDetected ? (
                <AlertTriangle className="w-10 h-10 text-white" />
              ) : (
                <CheckCircle className="w-10 h-10 text-white" />
              )}
            </div>
            
            <h2 className={`text-3xl sm:text-4xl font-bold mb-4 ${
              isDeceptionDetected ? 'text-red-400' : 'text-green-400'
            }`}>
              {result.decision}
            </h2>
            
            <div className="flex items-center justify-center space-x-2 mb-6">
              <BarChart3 className="w-5 h-5 text-slate-400" />
              <span className="text-xl text-slate-300">
                Confidence: <span className="font-bold text-white">{confidencePercentage}%</span>
              </span>
            </div>
            
            <p className="text-lg text-slate-400 max-w-2xl mx-auto">
              {result.reason}
            </p>
          </div>

          {/* Confidence Bar */}
          <div className="mb-8">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-slate-400">Confidence Level</span>
              <span className="text-sm font-medium text-white">{confidencePercentage}%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-3">
              <div 
                className={`h-3 rounded-full transition-all duration-1000 ease-out ${
                  result.confidence > 0.8 
                    ? 'bg-gradient-to-r from-green-500 to-green-400' 
                    : result.confidence > 0.6 
                    ? 'bg-gradient-to-r from-yellow-500 to-yellow-400'
                    : 'bg-gradient-to-r from-red-500 to-red-400'
                }`}
                style={{ width: `${result.confidence * 100}%` }}
              ></div>
            </div>
          </div>

          {/* Detailed Metrics */}
          {result.details && (
            <div className="grid sm:grid-cols-3 gap-6">
              <div className="bg-slate-700/50 rounded-xl p-6 text-center">
                <Mic className="w-8 h-8 text-blue-400 mx-auto mb-3" />
                <h4 className="text-lg font-semibold text-white mb-2">Vocal Analysis</h4>
                <div className="text-2xl font-bold text-blue-400 mb-1">
                  {(result.details.vocalStress * 100).toFixed(0)}%
                </div>
                <p className="text-sm text-slate-400">Stress indicators</p>
              </div>

              <div className="bg-slate-700/50 rounded-xl p-6 text-center">
                <Eye className="w-8 h-8 text-purple-400 mx-auto mb-3" />
                <h4 className="text-lg font-semibold text-white mb-2">Micro-Expressions</h4>
                <div className="text-2xl font-bold text-purple-400 mb-1">
                  {(result.details.microExpressions * 100).toFixed(0)}%
                </div>
                <p className="text-sm text-slate-400">Facial analysis</p>
              </div>

              <div className="bg-slate-700/50 rounded-xl p-6 text-center">
                <User className="w-8 h-8 text-green-400 mx-auto mb-3" />
                <h4 className="text-lg font-semibold text-white mb-2">Body Language</h4>
                <div className="text-2xl font-bold text-green-400 mb-1">
                  {(result.details.bodyLanguage * 100).toFixed(0)}%
                </div>
                <p className="text-sm text-slate-400">Behavioral cues</p>
              </div>
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link
            to="/capture"
            className="inline-flex items-center justify-center px-8 py-4 bg-gradient-to-r from-blue-600 to-blue-700 text-white font-semibold rounded-xl shadow-lg hover:shadow-blue-500/25 hover:shadow-xl transform hover:scale-105 transition-all duration-300 group"
          >
            <RotateCcw className="w-5 h-5 mr-3 group-hover:scale-110 transition-transform duration-200" />
            Run Another Analysis
          </Link>
          
          <Link
            to="/"
            className="inline-flex items-center justify-center px-8 py-4 border-2 border-slate-600 text-slate-300 font-semibold rounded-xl hover:border-slate-500 hover:text-white hover:bg-slate-700/50 transition-all duration-300 group"
          >
            <ArrowLeft className="w-5 h-5 mr-3 group-hover:scale-110 transition-transform duration-200" />
            Back to Home
          </Link>
        </div>

        {/* Disclaimer */}
        <div className="mt-12 bg-slate-800/30 backdrop-blur-sm border border-slate-700/30 rounded-2xl p-6">
          <div className="flex items-start space-x-3">
            <AlertTriangle className="w-5 h-5 text-yellow-500 mt-0.5 flex-shrink-0" />
            <div>
              <h4 className="text-lg font-semibold text-white mb-2">Important Disclaimer</h4>
              <p className="text-slate-400 text-sm leading-relaxed">
                This analysis is for demonstration purposes only. Results should not be used for legal, 
                employment, or other critical decisions. Always consult with qualified professionals 
                for important determinations.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ResultPage;