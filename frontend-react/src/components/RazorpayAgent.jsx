import React, { useState, useRef } from 'react';
import { Mic, X, MessageSquare, ChevronDown } from 'lucide-react';
import axios from 'axios';

export default function RazorpayAgent({ onCartUpdate }) {
  const [isOpen, setIsOpen] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [language, setLanguage] = useState('hi');
  const [messages, setMessages] = useState([
    { text: "Hello! Tap the mic to shop with voice.", sender: 'agent' }
  ]);
  const [pendingAction, setPendingAction] = useState(null);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const toggleChat = () => setIsOpen(!isOpen);

  const speakText = (text, lang) => {
    if (!('speechSynthesis' in window)) return;

    // Stop any ongoing speech
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang === 'hi' ? 'hi-IN' : lang === 'kn' ? 'kn-IN' : 'en-IN';

    // Slight adjustments for better sounding Indic voices
    utterance.rate = 0.9;
    utterance.pitch = 1.0;

    window.speechSynthesis.speak(utterance);
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const webmBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });

        try {
          const wavBlob = await convertWebMtoWAV(webmBlob);
          sendToBackend(wavBlob);
        } catch (e) {
          console.error("Audio conversion failed", e);
          sendToBackend(webmBlob); // Fallback
        }

        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      if (!isOpen) setIsOpen(true);

    } catch (err) {
      console.error("Mic access denied", err);
      setMessages(prev => [...prev, { text: "Microphone access denied. Please allow it in your browser.", sender: 'agent' }]);
    }
  };

  const stopRecording = () => {
    setIsRecording(false);
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
      setMessages(prev => [...prev, { text: "Listening & Transcribing...", sender: 'agent', isProcessing: true }]);
    }
  };

  const convertWebMtoWAV = async (webmBlob) => {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const arrayBuffer = await webmBlob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    const sampleRate = audioBuffer.sampleRate;
    const length = audioBuffer.length;
    const result = new Float32Array(length);
    audioBuffer.copyFromChannel(result, 0); // take first channel (mono)

    const buffer = new ArrayBuffer(44 + result.length * 2);
    const view = new DataView(buffer);

    const writeString = (view, offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + result.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, result.length * 2, true);

    let pcmOffset = 44;
    for (let i = 0; i < result.length; i++, pcmOffset += 2) {
      let s = Math.max(-1, Math.min(1, result[i]));
      view.setInt16(pcmOffset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([view], { type: 'audio/wav' });
  };

  const sendToBackend = async (blob, isConfirmation = false, pendingText = null) => {
    const formData = new FormData();
    if (blob) {
      formData.append('audio', blob, 'recording.wav');
    } else if (pendingText) {
      formData.append('text', pendingText);
    }
    formData.append('language', language);
    formData.append('phone_number', '+918888888888');
    
    // Add conversation context (last 4 messages)
    const recentContext = messages.slice(-4).map(m => `${m.sender}: ${m.text}`).join('\n');
    formData.append('context', recentContext);
    
    if (isConfirmation) {
      formData.append('skip_gate', 'true');
    }

    try {
      const res = await axios.post('http://127.0.0.1:8000/api/process-voice', formData);
      const data = res.data;

      // First, remove the processing message
      setMessages(prev => prev.filter(m => !m.isProcessing));

      if (data.status === "success") {
        const transcription = data.transcription || "Unknown";
        const agentRes = data.agent_response;

        // Show what the user said
        if (!isConfirmation) {
          setMessages(prev => [...prev, { text: transcription, sender: 'user' }]);
        } else {
          setMessages(prev => [...prev, { text: "Yes, confirm", sender: 'user' }]);
        }

        const isPending = agentRes.status && agentRes.status.startsWith('pending_');
        if (isPending && !isConfirmation) {
          setPendingAction({ transcription, language });
        } else {
          setPendingAction(null);
        }

        // Show agent response
        setMessages(prev => [...prev, {
          text: agentRes.message || "Done.",
          sender: 'agent',
          link: agentRes.payment_link || agentRes.invoice_id,
          isConfirmationRequired: isPending && !isConfirmation
        }]);

        // Speak the response using Web Speech API
        if (agentRes.message) {
          speakText(agentRes.message, language);
        }

        // Trigger cart refresh if onCartUpdate is provided
        if (onCartUpdate) {
          onCartUpdate();
        }
      } else {
        setMessages(prev => [...prev, {
          text: data.error || "An error occurred on the server.", sender: 'agent'
        }]);
      }

    } catch (err) {
      console.error(err);
      setMessages(prev => prev.filter(m => !m.isProcessing).concat({
        text: "Network Error while communicating with server.", sender: 'agent'
      }));
    }
  };

  return (
    <div className="agent-container">
      <div className={`agent-chat ${isOpen ? 'open' : 'hidden'}`}>
        <div className="chat-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <MessageSquare size={18} />
            <span>Razorpay Voice</span>
          </div>
          <select
            className="lang-select"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
          >
            <option value="en">English</option>
            <option value="hi">Hindi</option>
            <option value="kn">Kannada</option>
          </select>
          <X size={18} style={{ cursor: 'pointer' }} onClick={() => setIsOpen(false)} />
        </div>

        <div className="chat-messages">
          {messages.map((m, idx) => (
            <div key={idx} className={`msg msg-${m.sender} ${m.isProcessing ? 'pulsing' : ''}`}>
              {m.text}
              {m.link && typeof m.link === 'string' && m.link.startsWith('http') && (
                <div style={{ marginTop: '10px' }}>
                  <a href={m.link} target="_blank" rel="noreferrer" className="pay-btn">Proceed to Pay ↗</a>
                </div>
              )}
              {m.isConfirmationRequired && pendingAction && (
                <div style={{ marginTop: '10px' }}>
                  <button 
                    className="pay-btn" 
                    onClick={() => {
                      setMessages(prev => [...prev, { text: "Confirming...", sender: 'agent', isProcessing: true }]);
                      sendToBackend(null, true, pendingAction.transcription);
                    }}
                  >
                    Confirm
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="chat-input-area">
          <div className="input-placeholder">Tap to speak...</div>
          <button
            className={`mic-btn ${isRecording ? 'recording' : ''}`}
            onClick={isRecording ? stopRecording : startRecording}
          >
            <Mic size={24} />
          </button>
        </div>
      </div>

      {!isOpen && (
        <button className="floating-chat-btn" onClick={() => setIsOpen(true)}>
          <MessageSquare size={24} />
        </button>
      )}
    </div>
  );
}
