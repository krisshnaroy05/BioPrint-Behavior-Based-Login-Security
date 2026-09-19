import { useState, useRef, useCallback } from 'react';

export const useBiometrics = () => {
  const keyEventsRef = useRef([]);
  const mouseEventsRef = useRef([]);
  const lastMouseSampleRef = useRef(0);

  const handleKeyDown = useCallback((e) => {
    keyEventsRef.current.push({
      type: 'keydown',
      key: e.key,
      timestamp: performance.now()
    });
  }, []);

  const handleKeyUp = useCallback((e) => {
    keyEventsRef.current.push({
      type: 'keyup',
      key: e.key,
      timestamp: performance.now()
    });
  }, []);

  const handleMouseMove = useCallback((e) => {
    const now = performance.now();
    // Throttle mouse collection to every 25ms to preserve CPU
    if (now - lastMouseSampleRef.current > 25) {
      lastMouseSampleRef.current = now;
      mouseEventsRef.current.push({
        x: e.clientX,
        y: e.clientY,
        timestamp: now
      });
    }
  }, []);

  const resetTelemetry = useCallback(() => {
    keyEventsRef.current = [];
    mouseEventsRef.current = [];
  }, []);

  const getTelemetryPayload = useCallback((userId) => {
    return {
      user_id: userId,
      key_events: [...keyEventsRef.current],
      mouse_events: [...mouseEventsRef.current]
    };
  }, []);

  return {
    bindInput: {
      onKeyDown: handleKeyDown,
      onKeyUp: handleKeyUp
    },
    bindContainer: {
      onMouseMove: handleMouseMove
    },
    getTelemetryPayload,
    resetTelemetry
  };
};
