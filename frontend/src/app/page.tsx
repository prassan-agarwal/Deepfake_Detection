'use client';

import { useState, useRef, ReactNode } from 'react';
import { UploadCloud, FileVideo, ShieldAlert, ShieldCheck, Asterisk, XCircle, ArrowRight, Loader2, Play } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isHovering, setIsHovering] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [result, setResult] = useState<{ is_fake: boolean; probability: number; confidence: string; gradcam_base64?: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsHovering(true);
  };

  const handleDragLeave = () => {
    setIsHovering(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsHovering(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      validateAndSetFile(droppedFile);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const validateAndSetFile = (selectedFile: File) => {
    setError(null);
    setResult(null);
    if (!selectedFile.type.includes('video/mp4') && !selectedFile.name.endsWith('.mp4')) {
      setError("Please upload a valid MP4 video.");
      return;
    }
    setFile(selectedFile);
  };

  const resetState = () => {
    setFile(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const performDetection = async () => {
    if (!file) return;

    setIsUploading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('video', file);

    try {
      const response = await fetch('http://localhost:8000/api/detect', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Analysis failed. Make sure the backend server is running!");
      }

      const data = await response.json();
      setResult({
        is_fake: data.is_fake,
        probability: data.fake_probability,
        confidence: data.confidence_percentage,
        gradcam_base64: data.gradcam_base64,
      });
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred during detection.");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <main className="min-h-screen relative overflow-hidden flex flex-col items-center justify-center p-6">
      {/* Background Animated Orbs */}
      <div className="blob blob-1"></div>
      <div className="blob blob-2"></div>
      
      <div className="w-full max-w-4xl z-10 space-y-8 relative">
        {/* Header */}
        <div className="text-center space-y-4">
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass border border-blue-500/30 text-sm font-medium text-blue-400 mb-2"
          >
            <Asterisk size={16} className="animate-pulse" /> AI-Powered Spatiotemporal Frequency Analysis
          </motion.div>
          
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-5xl md:text-7xl font-black tracking-tight"
          >
            Detect <span className="text-gradient">Deepfakes</span><br/> in Seconds.
          </motion.h1>
          
          <motion.p 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="text-lg text-gray-400 max-w-2xl mx-auto"
          >
            Upload a video to our 3-Branch Hybrid Neural Network. We analyze spatial artifacts, temporal flickering, and frequency spectrums to reveal digital manipulation instantly.
          </motion.p>
        </div>

        {/* Main Interface */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="glass rounded-3xl p-8 backdrop-blur-2xl border-white/10"
        >
          <AnimatePresence mode="wait">
            {!file ? (
              // Upload State
              <motion.div 
                key="upload"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className={`relative border-2 border-dashed rounded-2xl p-12 flex flex-col items-center justify-center transition-all duration-300 ease-in-out cursor-pointer ${
                  isHovering ? 'border-blue-500 bg-blue-500/5' : 'border-gray-700 hover:border-gray-500 hover:bg-gray-800/30'
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input 
                  type="file" 
                  accept=".mp4,.avi,.mov" 
                  className="hidden" 
                  ref={fileInputRef}
                  onChange={handleFileChange}
                />
                <div className="bg-blue-500/20 p-4 rounded-full mb-6">
                  <UploadCloud size={48} className="text-blue-400" />
                </div>
                <h3 className="text-xl font-semibold mb-2 text-gray-200">Drag & Drop your video here</h3>
                <p className="text-sm text-gray-500 text-center max-w-sm">
                  Or click to browse. Supports MP4, AVI, and MOV files. We process the frames through ONNX-optimized models.
                </p>
              </motion.div>
            ) : (
              // Selected / Results State
              <motion.div 
                key="result"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="space-y-6"
              >
                <div className="flex items-center justify-between bg-black/40 p-4 rounded-2xl border border-white/5">
                  <div className="flex items-center gap-4">
                    <div className="bg-indigo-500/20 p-3 rounded-xl border border-indigo-500/30">
                      <FileVideo className="text-indigo-400" />
                    </div>
                    <div>
                      <h4 className="font-medium text-gray-200 truncate max-w-xs md:max-w-md">{file.name}</h4>
                      <p className="text-xs text-gray-500">{(file.size / (1024 * 1024)).toFixed(2)} MB • Ready for analysis</p>
                    </div>
                  </div>
                  {!isUploading && !result && (
                    <button onClick={resetState} className="text-gray-500 hover:text-red-400 transition-colors p-2">
                       <XCircle size={20} />
                    </button>
                  )}
                </div>

                {error && (
                  <div className="bg-red-500/10 border border-red-500/20 p-4 rounded-xl flex items-start gap-3 text-red-400">
                    <XCircle size={20} className="mt-0.5 shrink-0" />
                    <p className="text-sm">{error}</p>
                  </div>
                )}

                {!result && !error && (
                  <div className="flex justify-end gap-4 mt-6">
                    <button 
                      onClick={performDetection} 
                      disabled={isUploading}
                      className="bg-white text-black hover:bg-gray-200 hover:scale-105 active:scale-95 transition-all w-full md:w-auto px-8 py-4 rounded-xl font-semibold flex items-center justify-center gap-2 group disabled:opacity-70 disabled:pointer-events-none"
                    >
                      {isUploading ? (
                        <>
                          <Loader2 className="animate-spin" /> Processing Neural Network...
                        </>
                      ) : (
                        <>
                          Run Deepfake ML Scan <Play size={18} className="group-hover:translate-x-1 transition-transform" />
                        </>
                      )}
                    </button>
                  </div>
                )}

                {/* Highly Visual Results Dashboard */}
                {result && (
                  <motion.div 
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className={`mt-8 overflow-hidden rounded-2xl relative border ${result.is_fake ? 'border-red-500/50 bg-red-950/20' : 'border-emerald-500/50 bg-emerald-950/20'}`}
                  >
                     {/* Status Banner */}
                     <div className={`p-6 md:p-8 flex items-center gap-6 ${result.is_fake ? 'bg-gradient-to-r from-red-500/20 to-transparent' : 'bg-gradient-to-r from-emerald-500/20 to-transparent'}`}>
                        <div className={`p-4 rounded-full ${result.is_fake ? 'bg-red-500/20 text-red-400 border border-red-500/30 shadow-[0_0_30px_rgba(239,68,68,0.3)]' : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 flex items-center justify-center shadow-[0_0_30px_rgba(16,185,129,0.3)]'}`}>
                          {result.is_fake ? <ShieldAlert size={40} /> : <ShieldCheck size={40} />}
                        </div>
                        <div>
                           <h2 className="text-3xl font-bold tracking-tight mb-1 text-white">
                             {result.is_fake ? 'Synthetic Media Detected.' : 'Authentic Media Verified.'}
                           </h2>
                           <p className={`text-sm ${result.is_fake ? 'text-red-300' : 'text-emerald-300'}`}>
                             Our ensemble model analyzed spatial and temporal features across {result.probability > 0.9 ? 'high confidence' : 'detected'} thresholds.
                           </p>
                        </div>
                     </div>

                     <div className="px-8 pb-8 pt-4">
                        <div className="flex justify-between items-end mb-2">
                           <span className="text-sm font-medium text-gray-400 uppercase tracking-wider">Confidence Score</span>
                           <span className={`text-2xl font-bold ${result.is_fake ? 'text-red-400' : 'text-emerald-400'}`}>
                             {result.confidence}
                           </span>
                        </div>
                        
                        {/* Progress Bar */}
                        <div className="h-4 w-full bg-black/60 rounded-full overflow-hidden border border-white/5 relative">
                          <motion.div 
                             initial={{ width: 0 }}
                             animate={{ width: `${result.is_fake ? result.probability * 100 : (1 - result.probability) * 100}%` }}
                             transition={{ duration: 1, ease: "easeOut" }}
                             className={`h-full absolute left-0 top-0 ${result.is_fake ? 'bg-gradient-to-r from-red-600 to-rose-400' : 'bg-gradient-to-r from-emerald-600 to-teal-400'}`}
                          />
                        </div>

                        {/* GradCAM Heatmap UI element */}
                        {result.gradcam_base64 && (
                          <div className="mt-8 border-t border-white/10 pt-6">
                            <h3 className="text-sm font-medium text-gray-400 flex items-center gap-2 mb-4 uppercase tracking-wider">
                               <Asterisk size={14} className="text-blue-400" /> Model Attention Map
                            </h3>
                            <div className="rounded-xl overflow-hidden border border-white/10 shadow-lg bg-black/40 flex items-center justify-center p-2">
                               <img 
                                 src={result.gradcam_base64} 
                                 alt="AI Attention Heatmap" 
                                 className="rounded-lg object-contain w-full max-h-64"
                               />
                            </div>
                            <p className="text-xs text-gray-500 mt-3 text-center">
                              Highlighted regions indicate where the Vision Transformer focused its analysis.
                            </p>
                          </div>
                        )}

                        <div className="mt-8 flex justify-center">
                           <button onClick={resetState} className="text-sm px-4 py-2 rounded-full border border-white/20 hover:bg-white/10 transition-colors flex items-center gap-2">
                             Analyze Another Video <ArrowRight size={14} />
                           </button>
                        </div>
                     </div>
                  </motion.div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>
    </main>
  );
}
