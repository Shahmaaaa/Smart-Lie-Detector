import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, CheckCircle, AlertTriangle, Mic, Video } from 'lucide-react';

function ResultsPage() {
  const navigate = useNavigate();
  const location = useLocation();
  // Get the result object passed from the CapturePage's navigation state
  const { result } = location.state || {};

  // This handles the case where a user navigates to this page directly without any data
  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen text-center px-4">
        <h2 className="text-2xl font-bold text-white mb-4">No Analysis Data Found</h2>
        <p className="text-slate-400 mb-6">Please go back and analyze a recording first.</p>
        <button
          onClick={() => navigate('/')}
          className="inline-flex items-center px-6 py-3 border-2 border-slate-600 text-slate-300 font-semibold rounded-xl hover:border-slate-500 hover:text-white hover:bg-slate-700/50 transition-all"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Capture
        </button>
      </div>
    );
  }

  const isDeception = result.decision === 'Deception Indicated';

  return (
    <div className="px-4 sm:px-6 lg:px-8 py-12">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-3xl sm:text-4xl font-bold text-white mb-4">Analysis Results</h1>
          {/* Main Decision Display */}
          <div className={`inline-flex items-center space-x-3 px-6 py-3 rounded-full ${isDeception ? 'bg-red-500/10' : 'bg-green-500/10'}`}>
            {isDeception ? (
              <AlertTriangle className="w-8 h-8 text-red-400" />
            ) : (
              <CheckCircle className="w-8 h-8 text-green-400" />
            )}
            <span className={`text-2xl font-semibold ${isDeception ? 'text-red-400' : 'text-green-400'}`}>
              {result.decision}
            </span>
          </div>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-3xl p-8 space-y-6">
          {/* Overall Confidence Score */}
          <div>
            <h3 className="text-lg font-semibold text-slate-300 mb-2">Overall Confidence</h3>
            <p className="text-5xl font-bold text-white">{result.confidence}</p>
          </div>

          <div className="border-t border-slate-700"></div>

          {/* Detailed Breakdown Section */}
          <div>
            <h3 className="text-lg font-semibold text-slate-300 mb-4">Detailed Breakdown</h3>
            <div className="space-y-4">
              <div className="flex justify-between items-center bg-slate-900/50 p-4 rounded-lg">
                <div className="flex items-center space-x-3">
                  <Mic className="w-5 h-5 text-blue-400" />
                  <span className="text-slate-400">Vocal Analysis (Truth Probability)</span>
                </div>
                <span className="font-semibold text-white text-lg">{result.details.vocalEnsembleTruthProbability}</span>
              </div>
              <div className="flex justify-between items-center bg-slate-900/50 p-4 rounded-lg">
                <div className="flex items-center space-x-3">
                  <Video className="w-5 h-5 text-purple-400" />
                  <span className="text-slate-400">Facial Analysis (Truth Probability)</span>
                </div>
                <span className="font-semibold text-white text-lg">{result.details.facialAverageTruthProbability}</span>
              </div>
            </div>
          </div>
        </div>

        <div className="text-center mt-10">
            <button
                onClick={() => navigate('/')}
                className="inline-flex items-center px-6 py-3 border-2 border-slate-600 text-slate-300 font-semibold rounded-xl hover:border-slate-500 hover:text-white hover:bg-slate-700/50 transition-all duration-300"
            >
                <ArrowLeft className="w-4 h-4 mr-2" />
                Analyze Another
            </button>
        </div>
      </div>
    </div>
  );
}

export default ResultsPage;