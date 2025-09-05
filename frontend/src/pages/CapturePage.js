import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Camera, 
  Mic, 
  Play, 
  Square, 
  RotateCcw, 
  Zap, 
  CheckCircle,
  AlertCircle,
  Loader2
} from 'lucide-react';

function CapturePage() {
  const navigate = useNavigate();

  // --- State Management ---
  const [isRecording, setIsRecording] = useState(false);
  const [videoBlob, setVideoBlob] = useState(null);
  const [permissionGranted, setPermissionGranted] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);

  // --- Refs ---
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const durationIntervalRef = useRef(null);

  // --- Function to get camera permission ---
  const getCameraPermission = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: true, 
        video: { width: 1280, height: 720 } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setPermissionGranted(true);
    } catch (err) {
      console.error("Error getting media permissions.", err);
      alert("Could not access the camera and microphone. Please check your browser permissions.");
    }
  };

  // --- Recording Logic ---
  const handleStartRecording = () => {
    setVideoBlob(null);
    recordedChunksRef.current = [];
    setRecordingDuration(0);

    if (videoRef.current?.srcObject) {
      mediaRecorderRef.current = new MediaRecorder(videoRef.current.srcObject);

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
        setVideoBlob(blob);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);

      // Start duration counter
      durationIntervalRef.current = setInterval(() => {
        setRecordingDuration(prev => prev + 1);
      }, 1000);
    }
  };

  const handleStopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
    
    if (durationIntervalRef.current) {
      clearInterval(durationIntervalRef.current);
    }
  };

  const handleRetake = () => {
    setVideoBlob(null);
    setRecordingDuration(0);
  };

  // --- Analysis Logic ---
// --- Analysis Logic ---
  const handleAnalyze = async () => {
    if (!videoBlob) return;

    setIsAnalyzing(true);
    
    const formData = new FormData();
    formData.append('video', videoBlob, 'recording.webm');

    // This is the URL for your Django backend
const API_ENDPOINT = 'http://127.0.0.1:8000/api/analyze/';
    try {
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();
        navigate('/results', { state: { result: result } });

    } catch (error) {
        console.error("Error analyzing video:", error);
        alert("There was an error sending the video for analysis. Please make sure your backend server is running.");
    } finally {
        setIsAnalyzing(false);
    }
  };

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="px-4 sm:px-6 lg:px-8 py-12">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-3xl sm:text-4xl font-bold text-white mb-4">
            Capture & Analysis
          </h1>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto">
            Record your subject and let our AI analyze vocal patterns, micro-expressions, and behavioral indicators.
          </p>
        </div>

        {/* Main Recording Interface */}
        <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-3xl p-8 mb-8">
          {/* Camera Feed */}
          <div className="relative mb-8">
            <div className="aspect-video bg-slate-900 rounded-2xl overflow-hidden border-2 border-slate-600 relative">
              <video 
                ref={videoRef} 
                autoPlay 
                muted 
                className="w-full h-full object-cover"
              />
              
              {/* Recording Indicator */}
              {isRecording && (
                <div className="absolute top-4 left-4 flex items-center space-x-2 bg-red-600/90 backdrop-blur-sm px-3 py-2 rounded-lg">
                  <div className="w-3 h-3 bg-red-400 rounded-full animate-pulse"></div>
                  <span className="text-white font-medium">REC {formatDuration(recordingDuration)}</span>
                </div>
              )}

              {/* Permission Status */}
              {!permissionGranted && (
                <div className="absolute inset-0 bg-slate-900/90 flex items-center justify-center">
                  <div className="text-center">
                    <Camera className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                    <p className="text-slate-400 text-lg">Camera access required</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Controls */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            {!permissionGranted && (
              <button
                onClick={getCameraPermission}
                className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-green-600 to-green-700 text-white font-semibold rounded-xl shadow-lg hover:shadow-green-500/25 hover:shadow-xl transform hover:scale-105 transition-all duration-300 group"
              >
                <Camera className="w-5 h-5 mr-2 group-hover:scale-110 transition-transform duration-200" />
                Enable Camera & Mic
              </button>
            )}

            {permissionGranted && !isRecording && !videoBlob && (
              <button
                onClick={handleStartRecording}
                className="inline-flex items-center px-8 py-4 bg-gradient-to-r from-red-600 to-red-700 text-white font-semibold rounded-xl shadow-lg hover:shadow-red-500/25 hover:shadow-xl transform hover:scale-105 transition-all duration-300 group"
              >
                <Play className="w-6 h-6 mr-3 group-hover:scale-110 transition-transform duration-200" />
                Start Recording
              </button>
            )}

            {isRecording && (
              <button
                onClick={handleStopRecording}
                className="inline-flex items-center px-8 py-4 bg-gradient-to-r from-slate-600 to-slate-700 text-white font-semibold rounded-xl shadow-lg hover:shadow-slate-500/25 hover:shadow-xl transform hover:scale-105 transition-all duration-300 group"
              >
                <Square className="w-6 h-6 mr-3 group-hover:scale-110 transition-transform duration-200" />
                Stop Recording
              </button>
            )}
          </div>

          {/* Permission Status Indicators */}
          {permissionGranted && (
            <div className="flex justify-center space-x-6 mt-6">
              <div className="flex items-center space-x-2 text-green-400">
                <CheckCircle className="w-4 h-4" />
                <span className="text-sm font-medium">Camera Active</span>
              </div>
              <div className="flex items-center space-x-2 text-green-400">
                <Mic className="w-4 h-4" />
                <span className="text-sm font-medium">Microphone Active</span>
              </div>
            </div>
          )}
        </div>

        {/* Recording Preview & Analysis */}
        {videoBlob && (
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-3xl p-8">
            <h3 className="text-2xl font-semibold text-white mb-6 text-center">
              Recording Preview
            </h3>
            
            <div className="grid lg:grid-cols-2 gap-8 items-start">
              {/* Video Preview */}
              <div className="space-y-4">
                <div className="aspect-video bg-slate-900 rounded-xl overflow-hidden border border-slate-600">
                  <video 
                    src={URL.createObjectURL(videoBlob)} 
                    controls 
                    className="w-full h-full object-cover"
                  />
                </div>
                
                <button
                  onClick={handleRetake}
                  className="w-full inline-flex items-center justify-center px-4 py-3 border-2 border-slate-600 text-slate-300 font-medium rounded-xl hover:border-slate-500 hover:text-white hover:bg-slate-700/50 transition-all duration-300"
                >
                  <RotateCcw className="w-4 h-4 mr-2" />
                  Retake Recording
                </button>
              </div>

              {/* Analysis Section */}
              <div className="space-y-6">
                <div className="text-center">
                  <h4 className="text-xl font-semibold text-white mb-4">
                    Ready for Analysis
                  </h4>
                  <p className="text-slate-400 mb-6">
                    Our AI will analyze vocal patterns, micro-expressions, and behavioral indicators to detect deception.
                  </p>
                </div>

                <button
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
                  className="w-full inline-flex items-center justify-center px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl shadow-lg hover:shadow-blue-500/25 hover:shadow-xl transform hover:scale-105 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none group"
                >
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="w-6 h-6 mr-3 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Zap className="w-6 h-6 mr-3 group-hover:scale-110 transition-transform duration-200" />
                      Analyze Recording
                    </>
                  )}
                </button>

                {isAnalyzing && (
                  <div className="bg-slate-700/50 rounded-xl p-4">
                    <div className="flex items-center space-x-3 text-blue-400 mb-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="font-medium">Processing Analysis</span>
                    </div>
                    <div className="space-y-2 text-sm text-slate-400">
                      <p>• Analyzing vocal stress patterns...</p>
                      <p>• Detecting micro-expressions...</p>
                      <p>• Evaluating behavioral indicators...</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Instructions */}
        {!videoBlob && permissionGranted && (
          <div className="bg-slate-800/30 backdrop-blur-sm border border-slate-700/30 rounded-2xl p-6 mt-8">
            <div className="flex items-start space-x-3">
              <AlertCircle className="w-5 h-5 text-blue-400 mt-0.5 flex-shrink-0" />
              <div>
                <h4 className="text-lg font-semibold text-white mb-2">Recording Tips</h4>
                <ul className="text-slate-400 space-y-1">
                  <li>• Ensure good lighting on the subject's face</li>
                  <li>• Keep the camera stable and at eye level</li>
                  <li>• Record for at least 30 seconds for optimal analysis</li>
                  <li>• Ask clear questions and wait for complete responses</li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default CapturePage;