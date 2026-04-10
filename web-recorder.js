(function () {
    const TARGET_SAMPLE_RATE = 44100
    const SEGMENT_DURATION_SECONDS = 60
    const STUDY_TARGET_SECONDS = 10 * 60
    const EXPORT_FOLDER = "eval"
    const FIXED_MODE = "nose"
    const DEFAULT_LABEL = "silence"
    const PROCESSOR_BUFFER_SIZE = 1024

    const dom = {
        participantInput: document.getElementById("participant-code"),
        microphoneSelect: document.getElementById("microphone-select"),
        refreshButton: document.getElementById("refresh-microphones"),
        startMicTestButton: document.getElementById("start-microphone-test"),
        stopMicTestButton: document.getElementById("stop-microphone-test"),
        micTestLevelText: document.getElementById("microphone-test-level-text"),
        micTestLevelFill: document.getElementById("microphone-test-level-fill"),
        micTestStatus: document.getElementById("microphone-test-status"),
        micTestPlayback: document.getElementById("microphone-test-playback"),
        startButton: document.getElementById("start-recording"),
        stopButton: document.getElementById("stop-recording"),
        downloadZipButton: document.getElementById("download-zip"),
        clearSessionButton: document.getElementById("clear-session"),
        statusPill: document.getElementById("recorder-status-pill"),
        statusText: document.getElementById("recorder-status-text"),
        currentClassDisplay: document.getElementById("current-class-display"),
        savedCountDisplay: document.getElementById("saved-count-display"),
        segmentProgressText: document.getElementById("segment-progress-text"),
        segmentProgressFill: document.getElementById("segment-progress-fill"),
        sessionProgressText: document.getElementById("session-progress-text"),
        sessionProgressFill: document.getElementById("session-progress-fill"),
        message: document.getElementById("recorder-message"),
        savedList: document.getElementById("saved-list"),
        classButtons: Array.from(document.querySelectorAll(".class-button[data-label]")),
    }

    if (!dom.startButton) {
        return
    }

    const state = {
        audioContext: null,
        mediaStream: null,
        sourceNode: null,
        processorNode: null,
        isRecording: false,
        currentLabel: DEFAULT_LABEL,
        inputSampleRate: TARGET_SAMPLE_RATE,
        targetInputSamples: TARGET_SAMPLE_RATE * SEGMENT_DURATION_SECONDS,
        segmentChunks: [],
        segmentInputSamples: 0,
        segmentEvents: [{ label: DEFAULT_LABEL, sample: 0 }],
        activeConfig: null,
        sessionSegments: [],
        totalSavedSeconds: 0,
        segmentSequence: 0,
        downloadInProgress: false,
        lastUiRefreshAt: 0,
        isMicTestActive: false,
        isMicTestStopping: false,
        micTestStream: null,
        micTestRecorder: null,
        micTestChunks: [],
        micTestAudioUrl: "",
        micTestAudioContext: null,
        micTestAnalyser: null,
        micTestSourceNode: null,
        micTestMeterFrame: 0,
    }

    const formControls = [
        dom.participantInput,
        dom.microphoneSelect,
        dom.refreshButton,
    ]

    function getSupportBlocker() {
        if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== "function") {
            return "This browser does not support microphone recording."
        }

        if (!(window.AudioContext || window.webkitAudioContext)) {
            return "This browser does not support the Web Audio features required for WAV export."
        }

        const isLocalHost = location.hostname === "localhost" || location.hostname === "127.0.0.1"
        if (!window.isSecureContext && !isLocalHost) {
            return "Microphone access requires HTTPS or localhost."
        }

        return null
    }

    function getMicTestSupportBlocker() {
        if (typeof window.MediaRecorder !== "function") {
            return "This browser cannot save a microphone test clip for playback."
        }

        return null
    }

    function setMessage(text, tone) {
        dom.message.textContent = text
        dom.message.dataset.tone = tone
    }

    function setMicrophoneTestStatus(text) {
        dom.micTestStatus.textContent = text
    }

    function getSelectedMicrophoneLabel() {
        const selectedOption = dom.microphoneSelect.selectedOptions[0]
        return selectedOption ? selectedOption.textContent : "System default microphone"
    }

    function getFormConfig() {
        return {
            participantCode: dom.participantInput.value.trim(),
            deviceId: dom.microphoneSelect.value || "default",
            deviceLabel: getSelectedMicrophoneLabel(),
        }
    }

    function sanitizeToken(value, fallback) {
        const normalized = value
            .normalize("NFKD")
            .replace(/[^\w-]+/g, "_")
            .replace(/_+/g, "_")
            .replace(/^_+|_+$/g, "")

        return normalized || fallback
    }

    function formatTimestamp(date) {
        const pad = (value) => String(value).padStart(2, "0")
        return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}_${pad(date.getHours())}-${pad(date.getMinutes())}-${pad(date.getSeconds())}`
    }

    function formatDuration(totalSeconds) {
        const clampedSeconds = Math.max(0, Math.floor(totalSeconds))
        const hours = Math.floor(clampedSeconds / 3600)
        const minutes = Math.floor((clampedSeconds % 3600) / 60)
        const seconds = clampedSeconds % 60

        if (hours > 0) {
            return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`
        }

        return `${minutes}:${String(seconds).padStart(2, "0")}`
    }

    function setFormDisabled(disabled) {
        formControls.forEach((control) => {
            control.disabled = disabled
        })
    }

    function clearMicrophoneTestSample() {
        if (state.micTestAudioUrl) {
            URL.revokeObjectURL(state.micTestAudioUrl)
            state.micTestAudioUrl = ""
        }

        dom.micTestPlayback.pause()
        dom.micTestPlayback.removeAttribute("src")
        dom.micTestPlayback.hidden = true
        dom.micTestPlayback.load()
    }

    function resetMicrophoneTestMeter(text) {
        dom.micTestLevelFill.style.width = "0%"
        dom.micTestLevelText.textContent = text
    }

    function ensureSegmentUrls(segment) {
        if (!segment.wavUrl) {
            segment.wavUrl = URL.createObjectURL(segment.wavBlob)
        }

        if (!segment.csvUrl) {
            const csvBlob = new Blob([segment.csvText], { type: "text/csv;charset=utf-8" })
            segment.csvUrl = URL.createObjectURL(csvBlob)
        }
    }

    function revokeSegmentUrls(segment) {
        if (segment.wavUrl) {
            URL.revokeObjectURL(segment.wavUrl)
            segment.wavUrl = ""
        }

        if (segment.csvUrl) {
            URL.revokeObjectURL(segment.csvUrl)
            segment.csvUrl = ""
        }
    }

    function renderSavedList() {
        dom.savedList.innerHTML = ""

        if (!state.sessionSegments.length) {
            const emptyState = document.createElement("div")
            emptyState.className = "saved-empty"
            emptyState.textContent = "No saved segments yet. WAV and CSV files will appear here after the first completed save."
            dom.savedList.append(emptyState)
            return
        }

        const segmentsToRender = [...state.sessionSegments].reverse()
        segmentsToRender.forEach((segment) => {
            ensureSegmentUrls(segment)

            const item = document.createElement("div")
            item.className = "saved-item"

            const title = document.createElement("strong")
            title.textContent = `${segment.exportFolder}/raw/${segment.wavName}`
            item.append(title)

            const meta = document.createElement("div")
            meta.className = "saved-meta"
            meta.innerHTML = `
                <span>${segment.partial ? "partial" : "full"}</span>
                <span>${formatDuration(segment.durationSeconds)}</span>
                <span>${segment.labelRowCount} label rows</span>
                <span>${segment.exportFolder}/label/${segment.csvName}</span>
            `
            item.append(meta)

            const links = document.createElement("div")
            links.className = "saved-links"
            links.innerHTML = `
                <a class="saved-link" href="${segment.wavUrl}" download="${segment.wavName}">Download WAV</a>
                <a class="saved-link" href="${segment.csvUrl}" download="${segment.csvName}">Download CSV</a>
            `
            item.append(links)

            dom.savedList.append(item)
        })
    }

    function getCurrentSegmentSeconds() {
        if (!state.segmentInputSamples || !state.inputSampleRate) {
            return 0
        }

        return state.segmentInputSamples / state.inputSampleRate
    }

    function updateMicrophoneTestUi() {
        const blocker = getSupportBlocker() || getMicTestSupportBlocker()
        dom.startMicTestButton.disabled = Boolean(blocker) || state.isRecording || state.isMicTestActive || state.isMicTestStopping
        dom.stopMicTestButton.disabled = !state.isMicTestActive || state.isMicTestStopping
        dom.micTestPlayback.hidden = !state.micTestAudioUrl

        if (blocker && !state.isMicTestActive && !state.isMicTestStopping) {
            setMicrophoneTestStatus(blocker)
            resetMicrophoneTestMeter("Unavailable")
        }
    }

    function updateStatus() {
        const blocker = getSupportBlocker()
        const currentSegmentSeconds = getCurrentSegmentSeconds()
        const totalVisibleSeconds = Math.min(
            STUDY_TARGET_SECONDS,
            state.totalSavedSeconds + (state.isRecording ? currentSegmentSeconds : 0)
        )
        const savedSegmentsLabel = state.sessionSegments.length === 1 ? "1 saved" : `${state.sessionSegments.length} saved`

        dom.currentClassDisplay.textContent = state.isRecording ? state.currentLabel.toUpperCase() : "READY"
        dom.savedCountDisplay.textContent = `${savedSegmentsLabel} · ${formatDuration(state.totalSavedSeconds)}`

        dom.segmentProgressText.textContent = `${formatDuration(currentSegmentSeconds)} / ${formatDuration(SEGMENT_DURATION_SECONDS)}`
        dom.segmentProgressFill.style.width = `${Math.min(100, (currentSegmentSeconds / SEGMENT_DURATION_SECONDS) * 100)}%`

        dom.sessionProgressText.textContent = `${formatDuration(totalVisibleSeconds)} / ${formatDuration(STUDY_TARGET_SECONDS)}`
        dom.sessionProgressFill.style.width = `${Math.min(100, (totalVisibleSeconds / STUDY_TARGET_SECONDS) * 100)}%`

        dom.startButton.disabled = Boolean(blocker) || state.isRecording || state.isMicTestActive || state.isMicTestStopping
        dom.stopButton.disabled = !state.isRecording
        dom.downloadZipButton.disabled = state.isRecording || state.isMicTestActive || state.isMicTestStopping || !state.sessionSegments.length || !window.JSZip || state.downloadInProgress
        dom.clearSessionButton.disabled = state.isRecording || state.isMicTestActive || state.isMicTestStopping || !state.sessionSegments.length

        dom.classButtons.forEach((button) => {
            button.disabled = !state.isRecording
            button.dataset.active = String(button.dataset.label === state.currentLabel)
        })

        if (blocker) {
            dom.statusPill.className = "status-pill blocked"
            dom.statusPill.textContent = "Blocked"
            dom.statusText.textContent = blocker
            if (!state.isRecording && !state.isMicTestActive && !state.isMicTestStopping) {
                setMessage(blocker, "error")
            }
        } else if (state.isRecording && state.activeConfig) {
            dom.statusPill.className = "status-pill live"
            dom.statusPill.textContent = "Recording"
            dom.statusText.textContent = `NOSE ONLY • ${formatDuration(SEGMENT_DURATION_SECONDS)} segments • ${state.activeConfig.deviceLabel}`
        } else if (state.isMicTestActive || state.isMicTestStopping) {
            dom.statusPill.className = "status-pill live"
            dom.statusPill.textContent = "Mic Check"
            dom.statusText.textContent = state.isMicTestStopping
                ? "Stopping microphone test..."
                : `Testing microphone • ${getSelectedMicrophoneLabel()}`
        } else {
            dom.statusPill.className = "status-pill ready"
            dom.statusPill.textContent = "Ready"
            dom.statusText.textContent = "Click Start Recording to begin the session."
        }
    }

    function refreshUi(force) {
        const now = performance.now()
        if (!force && now - state.lastUiRefreshAt < 100) {
            return
        }

        state.lastUiRefreshAt = now
        updateStatus()
        updateMicrophoneTestUi()
    }

    function populateMicrophoneOptions(inputDevices) {
        const previousSelection = dom.microphoneSelect.value || "default"
        dom.microphoneSelect.innerHTML = ""

        const defaultOption = document.createElement("option")
        defaultOption.value = "default"
        defaultOption.textContent = "System default microphone"
        dom.microphoneSelect.append(defaultOption)

        inputDevices.forEach((device, index) => {
            const option = document.createElement("option")
            option.value = device.deviceId
            option.textContent = device.label || `Microphone ${index + 1}`
            dom.microphoneSelect.append(option)
        })

        const hasPreviousSelection = inputDevices.some((device) => device.deviceId === previousSelection)
        dom.microphoneSelect.value = hasPreviousSelection ? previousSelection : "default"
    }

    async function refreshMicrophones(options) {
        const settings = options || {}
        const blocker = getSupportBlocker()
        if (blocker) {
            refreshUi(true)
            return
        }

        if (settings.requestPermission) {
            const previewStream = await navigator.mediaDevices.getUserMedia({ audio: true })
            previewStream.getTracks().forEach((track) => track.stop())
        }

        const devices = await navigator.mediaDevices.enumerateDevices()
        const inputDevices = devices.filter((device) => device.kind === "audioinput")
        populateMicrophoneOptions(inputDevices)
        refreshUi(true)
    }

    function buildAudioConstraints(deviceId) {
        const audioConstraints = {
            channelCount: 1,
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
        }

        if (deviceId && deviceId !== "default") {
            audioConstraints.deviceId = { exact: deviceId }
        }

        return {
            audio: audioConstraints,
            video: false,
        }
    }

    async function openMicrophone(deviceId) {
        try {
            return await navigator.mediaDevices.getUserMedia(buildAudioConstraints(deviceId))
        } catch (error) {
            if (deviceId && deviceId !== "default") {
                dom.microphoneSelect.value = "default"
                setMessage("The selected microphone was not available. Falling back to the system default device.", "warning")
                return navigator.mediaDevices.getUserMedia(buildAudioConstraints("default"))
            }

            throw error
        }
    }

    function getMediaRecorderOptions() {
        if (typeof window.MediaRecorder !== "function") {
            return null
        }

        const mimeCandidates = [
            "audio/webm;codecs=opus",
            "audio/webm",
            "audio/ogg;codecs=opus",
            "audio/ogg",
        ]

        if (typeof window.MediaRecorder.isTypeSupported !== "function") {
            return {}
        }

        for (const candidate of mimeCandidates) {
            if (window.MediaRecorder.isTypeSupported(candidate)) {
                return { mimeType: candidate }
            }
        }

        return {}
    }

    async function setupMicrophoneTestMeter(stream) {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext
        const audioContext = new AudioContextClass()
        await audioContext.resume()

        const analyser = audioContext.createAnalyser()
        analyser.fftSize = 2048
        analyser.smoothingTimeConstant = 0.85

        const sourceNode = audioContext.createMediaStreamSource(stream)
        sourceNode.connect(analyser)

        state.micTestAudioContext = audioContext
        state.micTestAnalyser = analyser
        state.micTestSourceNode = sourceNode

        const data = new Uint8Array(analyser.fftSize)
        const step = function () {
            if ((!state.isMicTestActive && !state.isMicTestStopping) || !state.micTestAnalyser) {
                return
            }

            state.micTestAnalyser.getByteTimeDomainData(data)

            let sumSquares = 0
            for (let index = 0; index < data.length; index += 1) {
                const centeredSample = (data[index] - 128) / 128
                sumSquares += centeredSample * centeredSample
            }

            const rms = Math.sqrt(sumSquares / data.length)
            const normalizedLevel = Math.min(1, rms * 7)
            const fillWidth = normalizedLevel > 0 ? Math.max(4, normalizedLevel * 100) : 0
            dom.micTestLevelFill.style.width = `${fillWidth}%`

            if (normalizedLevel < 0.05) {
                dom.micTestLevelText.textContent = "No signal"
            } else if (normalizedLevel < 0.16) {
                dom.micTestLevelText.textContent = "Low signal"
            } else if (normalizedLevel < 0.32) {
                dom.micTestLevelText.textContent = "Good signal"
            } else {
                dom.micTestLevelText.textContent = "Strong signal"
            }

            state.micTestMeterFrame = window.requestAnimationFrame(step)
        }

        step()
    }

    async function teardownMicrophoneTestMeter() {
        if (state.micTestMeterFrame) {
            window.cancelAnimationFrame(state.micTestMeterFrame)
            state.micTestMeterFrame = 0
        }

        if (state.micTestSourceNode) {
            state.micTestSourceNode.disconnect()
            state.micTestSourceNode = null
        }

        state.micTestAnalyser = null

        if (state.micTestAudioContext) {
            await state.micTestAudioContext.close()
            state.micTestAudioContext = null
        }
    }

    async function finalizeMicrophoneTestCapture() {
        const recorder = state.micTestRecorder
        const mimeType = recorder && recorder.mimeType ? recorder.mimeType : "audio/webm"
        const chunks = state.micTestChunks.slice()

        state.micTestRecorder = null
        state.micTestChunks = []
        state.isMicTestActive = false
        state.isMicTestStopping = false

        if (state.micTestStream) {
            state.micTestStream.getTracks().forEach((track) => track.stop())
            state.micTestStream = null
        }

        await teardownMicrophoneTestMeter()

        if (!state.isRecording) {
            setFormDisabled(false)
        }

        if (chunks.length) {
            const testBlob = new Blob(chunks, { type: mimeType })
            if (testBlob.size > 0) {
                clearMicrophoneTestSample()
                state.micTestAudioUrl = URL.createObjectURL(testBlob)
                dom.micTestPlayback.src = state.micTestAudioUrl
                dom.micTestPlayback.hidden = false
                dom.micTestPlayback.load()
                setMicrophoneTestStatus("Test clip ready. Listen back and confirm that the selected microphone sounds correct.")
                resetMicrophoneTestMeter("Ready")
            } else {
                setMicrophoneTestStatus("The microphone test finished, but no audio was captured. Try again and speak closer to the microphone.")
                resetMicrophoneTestMeter("No signal")
            }
        } else {
            setMicrophoneTestStatus("The microphone test finished, but no audio was captured. Try again.")
            resetMicrophoneTestMeter("No signal")
        }

        refreshUi(true)
    }

    async function startMicrophoneTest() {
        const blocker = getSupportBlocker() || getMicTestSupportBlocker()
        if (blocker || state.isRecording || state.isMicTestActive || state.isMicTestStopping) {
            refreshUi(true)
            return
        }

        const formConfig = getFormConfig()
        dom.micTestPlayback.pause()
        setMicrophoneTestStatus(`Requesting microphone access for ${formConfig.deviceLabel}...`)

        try {
            const stream = await openMicrophone(formConfig.deviceId)
            const recorderOptions = getMediaRecorderOptions()
            const recorder = recorderOptions ? new MediaRecorder(stream, recorderOptions) : new MediaRecorder(stream)

            clearMicrophoneTestSample()
            state.micTestStream = stream
            state.micTestRecorder = recorder
            state.micTestChunks = []
            state.isMicTestActive = true
            state.isMicTestStopping = false

            setFormDisabled(true)
            await setupMicrophoneTestMeter(stream)

            recorder.ondataavailable = function (event) {
                if (event.data && event.data.size > 0) {
                    state.micTestChunks.push(event.data)
                }
            }

            recorder.onstop = function () {
                finalizeMicrophoneTestCapture().catch((error) => {
                    console.error(error)
                    setMicrophoneTestStatus("The microphone test could not be finalized. Please try again.")
                    refreshUi(true)
                })
            }

            recorder.start()
            setMicrophoneTestStatus(`Testing ${getSelectedMicrophoneLabel()}. Speak for a few seconds, then stop and listen back.`)
            refreshUi(true)
        } catch (error) {
            if (state.micTestStream) {
                state.micTestStream.getTracks().forEach((track) => track.stop())
                state.micTestStream = null
            }

            await teardownMicrophoneTestMeter()
            state.isMicTestActive = false
            state.isMicTestStopping = false
            setFormDisabled(false)

            const errorMessage = error && error.name === "NotAllowedError"
                ? "Microphone access was blocked. Allow the microphone and try again."
                : "The microphone test could not start. Check browser permissions and device selection."
            setMicrophoneTestStatus(errorMessage)
            console.error(error)
            refreshUi(true)
        }
    }

    async function stopMicrophoneTest() {
        if (!state.isMicTestActive || !state.micTestRecorder || state.isMicTestStopping) {
            return
        }

        state.isMicTestStopping = true
        setMicrophoneTestStatus("Stopping microphone test...")
        refreshUi(true)

        if (state.micTestRecorder.state === "inactive") {
            await finalizeMicrophoneTestCapture()
            return
        }

        state.micTestRecorder.stop()
    }

    function mergeFloatChunks(chunks, totalSamples) {
        const merged = new Float32Array(totalSamples)
        let offset = 0

        chunks.forEach((chunk) => {
            merged.set(chunk, offset)
            offset += chunk.length
        })

        return merged
    }

    function resampleFloat32(inputSamples, inputRate, outputRate) {
        if (!inputSamples.length) {
            return new Float32Array(0)
        }

        if (inputRate === outputRate) {
            return inputSamples
        }

        const ratio = inputRate / outputRate
        const outputLength = Math.max(1, Math.round(inputSamples.length / ratio))
        const outputSamples = new Float32Array(outputLength)

        for (let outputIndex = 0; outputIndex < outputLength; outputIndex += 1) {
            const sourceIndex = outputIndex * ratio
            const lowerIndex = Math.floor(sourceIndex)
            const upperIndex = Math.min(inputSamples.length - 1, lowerIndex + 1)
            const interpolation = sourceIndex - lowerIndex

            outputSamples[outputIndex] = inputSamples[lowerIndex] * (1 - interpolation) + inputSamples[upperIndex] * interpolation
        }

        return outputSamples
    }

    function createWavBlob(samples, sampleRate) {
        const bytesPerSample = 2
        const blockAlign = bytesPerSample
        const byteRate = sampleRate * blockAlign
        const dataSize = samples.length * bytesPerSample
        const buffer = new ArrayBuffer(44 + dataSize)
        const view = new DataView(buffer)

        function writeString(offset, value) {
            for (let index = 0; index < value.length; index += 1) {
                view.setUint8(offset + index, value.charCodeAt(index))
            }
        }

        writeString(0, "RIFF")
        view.setUint32(4, 36 + dataSize, true)
        writeString(8, "WAVE")
        writeString(12, "fmt ")
        view.setUint32(16, 16, true)
        view.setUint16(20, 1, true)
        view.setUint16(22, 1, true)
        view.setUint32(24, sampleRate, true)
        view.setUint32(28, byteRate, true)
        view.setUint16(32, blockAlign, true)
        view.setUint16(34, 16, true)
        writeString(36, "data")
        view.setUint32(40, dataSize, true)

        let offset = 44
        for (let sampleIndex = 0; sampleIndex < samples.length; sampleIndex += 1) {
            const sample = Math.max(-1, Math.min(1, samples[sampleIndex]))
            const int16Sample = sample < 0 ? sample * 0x8000 : sample * 0x7fff
            view.setInt16(offset, int16Sample, true)
            offset += bytesPerSample
        }

        return new Blob([buffer], { type: "audio/wav" })
    }

    function buildCsvData(events, inputRate, outputRate, totalOutputSamples) {
        const rawEvents = events.length ? events : [{ label: DEFAULT_LABEL, sample: 0 }]
        const convertedEvents = rawEvents.map((event) => ({
            label: event.label,
            sample: Math.max(0, Math.min(totalOutputSamples, Math.round((event.sample / inputRate) * outputRate))),
        }))

        if (!convertedEvents.length || convertedEvents[0].sample !== 0) {
            convertedEvents.unshift({
                label: convertedEvents[0] ? convertedEvents[0].label : DEFAULT_LABEL,
                sample: 0,
            })
        }

        const normalizedEvents = []
        convertedEvents.forEach((event) => {
            if (!normalizedEvents.length) {
                normalizedEvents.push(event)
                return
            }

            const lastEvent = normalizedEvents[normalizedEvents.length - 1]
            if (event.sample === lastEvent.sample) {
                lastEvent.label = event.label
                return
            }

            if (event.label === lastEvent.label) {
                return
            }

            normalizedEvents.push(event)
        })

        const lines = ["class,start_sample,end_sample"]
        normalizedEvents.forEach((event, index) => {
            const nextEvent = normalizedEvents[index + 1]
            const endSample = nextEvent ? nextEvent.sample : totalOutputSamples
            lines.push(`${event.label},${event.sample},${Math.max(event.sample, endSample)}`)
        })

        return {
            text: lines.join("\n"),
            labelRowCount: normalizedEvents.length,
        }
    }

    function resetSegmentState() {
        state.segmentChunks = []
        state.segmentInputSamples = 0
        state.segmentEvents = [{ label: state.currentLabel, sample: 0 }]
    }

    function finalizeCurrentSegment(partial) {
        if (!state.segmentInputSamples) {
            return
        }

        const mergedSamples = mergeFloatChunks(state.segmentChunks, state.segmentInputSamples)
        const resampledSamples = resampleFloat32(mergedSamples, state.inputSampleRate, TARGET_SAMPLE_RATE)
        const csvData = buildCsvData(state.segmentEvents, state.inputSampleRate, TARGET_SAMPLE_RATE, resampledSamples.length)
        const wavBlob = createWavBlob(resampledSamples, TARGET_SAMPLE_RATE)

        state.segmentSequence += 1

        const timestamp = formatTimestamp(new Date())
        const sequence = String(state.segmentSequence).padStart(2, "0")
        const config = state.activeConfig || getFormConfig()
        const participantToken = sanitizeToken(config.participantCode, "unknown_participant")
        const basename = `${participantToken}_${FIXED_MODE}_${timestamp}_${sequence}`

        state.sessionSegments.push({
            participantCode: config.participantCode,
            exportFolder: EXPORT_FOLDER,
            wavName: `${basename}.wav`,
            csvName: `${basename}.csv`,
            wavBlob,
            csvText: csvData.text,
            wavUrl: "",
            csvUrl: "",
            durationSeconds: resampledSamples.length / TARGET_SAMPLE_RATE,
            partial,
            labelRowCount: csvData.labelRowCount,
        })

        state.totalSavedSeconds += resampledSamples.length / TARGET_SAMPLE_RATE
        renderSavedList()
        refreshUi(true)
    }

    function pushChunk(inputData) {
        let offset = 0

        while (offset < inputData.length) {
            const remainingSamples = state.targetInputSamples - state.segmentInputSamples
            const chunkLength = Math.min(remainingSamples, inputData.length - offset)
            const copiedChunk = new Float32Array(chunkLength)
            copiedChunk.set(inputData.subarray(offset, offset + chunkLength))
            state.segmentChunks.push(copiedChunk)
            state.segmentInputSamples += chunkLength
            offset += chunkLength

            if (state.segmentInputSamples === state.targetInputSamples) {
                finalizeCurrentSegment(false)
                if (!state.isRecording) {
                    return
                }
                resetSegmentState()
            }
        }
    }

    function handleAudioProcess(audioEvent) {
        const outputChannel = audioEvent.outputBuffer.getChannelData(0)
        outputChannel.fill(0)

        if (!state.isRecording) {
            return
        }

        const inputChannel = audioEvent.inputBuffer.getChannelData(0)
        pushChunk(inputChannel)
        refreshUi(false)
    }

    async function teardownCapture() {
        if (state.processorNode) {
            state.processorNode.onaudioprocess = null
            state.processorNode.disconnect()
            state.processorNode = null
        }

        if (state.sourceNode) {
            state.sourceNode.disconnect()
            state.sourceNode = null
        }

        if (state.mediaStream) {
            state.mediaStream.getTracks().forEach((track) => track.stop())
            state.mediaStream = null
        }

        if (state.audioContext) {
            await state.audioContext.close()
            state.audioContext = null
        }
    }

    async function startRecording() {
        const blocker = getSupportBlocker()
        if (blocker || state.isRecording) {
            refreshUi(true)
            return
        }

        if (state.isMicTestActive || state.isMicTestStopping) {
            setMessage("Stop the microphone test before starting the full recording.", "warning")
            refreshUi(true)
            return
        }

        const formConfig = getFormConfig()
        if (!formConfig.participantCode) {
            setMessage("Enter a participant code before starting the recording.", "error")
            refreshUi(true)
            return
        }

        dom.micTestPlayback.pause()
        setMessage("Requesting microphone access and preparing the recorder...", "info")

        try {
            const mediaStream = await openMicrophone(formConfig.deviceId)
            const AudioContextClass = window.AudioContext || window.webkitAudioContext
            const audioContext = new AudioContextClass()
            await audioContext.resume()

            state.mediaStream = mediaStream
            state.audioContext = audioContext
            state.inputSampleRate = audioContext.sampleRate
            state.targetInputSamples = Math.round(audioContext.sampleRate * SEGMENT_DURATION_SECONDS)
            state.activeConfig = {
                ...formConfig,
                deviceLabel: getSelectedMicrophoneLabel(),
            }
            state.isRecording = true
            state.currentLabel = DEFAULT_LABEL

            resetSegmentState()

            state.sourceNode = audioContext.createMediaStreamSource(mediaStream)
            state.processorNode = audioContext.createScriptProcessor(PROCESSOR_BUFFER_SIZE, 1, 1)
            state.processorNode.onaudioprocess = handleAudioProcess
            state.sourceNode.connect(state.processorNode)
            state.processorNode.connect(audioContext.destination)

            setFormDisabled(true)
            setMessage("Recording is live. Use the buttons or W, E, and R to label each breathing phase.", "info")

            try {
                await refreshMicrophones()
            } catch (deviceError) {
                console.warn(deviceError)
            }

            refreshUi(true)
        } catch (error) {
            await teardownCapture()
            const errorMessage = error && error.name === "NotAllowedError"
                ? "Microphone access was blocked. Allow the microphone and try again."
                : "The recorder could not start the microphone. Check browser permissions and device selection."
            setMessage(errorMessage, "error")
            console.error(error)
            refreshUi(true)
        }
    }

    function setCurrentLabel(nextLabel) {
        if (!state.isRecording || nextLabel === state.currentLabel) {
            return
        }

        const lastEvent = state.segmentEvents[state.segmentEvents.length - 1]
        if (lastEvent && lastEvent.sample === state.segmentInputSamples) {
            lastEvent.label = nextLabel
        } else {
            state.segmentEvents.push({
                label: nextLabel,
                sample: state.segmentInputSamples,
            })
        }

        state.currentLabel = nextLabel
        refreshUi(true)
    }

    async function stopRecording() {
        if (!state.isRecording) {
            return
        }

        state.isRecording = false
        setFormDisabled(false)

        const hasPartialSegment = state.segmentInputSamples > 0

        await teardownCapture()

        if (hasPartialSegment) {
            finalizeCurrentSegment(state.segmentInputSamples < state.targetInputSamples)
        }

        state.activeConfig = null
        resetSegmentState()
        setMessage("Recording stopped. You can download the ZIP or start another session.", "info")
        refreshUi(true)
    }

    function triggerBlobDownload(blob, filename) {
        const url = URL.createObjectURL(blob)
        const anchor = document.createElement("a")
        anchor.href = url
        anchor.download = filename
        document.body.append(anchor)
        anchor.click()
        anchor.remove()
        window.setTimeout(() => URL.revokeObjectURL(url), 1000)
    }

    async function downloadZip() {
        if (!state.sessionSegments.length || state.downloadInProgress) {
            return
        }

        if (!window.JSZip) {
            setMessage("The ZIP library did not load correctly. Refresh the page and try again.", "error")
            refreshUi(true)
            return
        }

        state.downloadInProgress = true
        setMessage("Preparing the ZIP package...", "info")
        refreshUi(true)

        try {
            const zip = new window.JSZip()

            state.sessionSegments.forEach((segment) => {
                zip.file(`${segment.exportFolder}/raw/${segment.wavName}`, segment.wavBlob)
                zip.file(`${segment.exportFolder}/label/${segment.csvName}`, segment.csvText)
            })

            const summaryLines = [
                "Breathing Recorder export",
                `Created: ${new Date().toISOString()}`,
                `Segments: ${state.sessionSegments.length}`,
                `Total duration seconds: ${state.totalSavedSeconds.toFixed(2)}`,
                `Mode: ${FIXED_MODE.toUpperCase()}`,
                `Segment length seconds: ${SEGMENT_DURATION_SECONDS}`,
                "",
                ...state.sessionSegments.map((segment) => `${segment.exportFolder}/${segment.wavName} | ${segment.durationSeconds.toFixed(2)}s | ${segment.partial ? "partial" : "full"}`),
            ]
            zip.file("session-summary.txt", summaryLines.join("\n"))

            const zipBlob = await zip.generateAsync({
                type: "blob",
                compression: "DEFLATE",
                compressionOptions: { level: 6 },
            })

            const participantToken = sanitizeToken(state.sessionSegments[0].participantCode, "breathing_session")
            const zipName = `${participantToken}_${FIXED_MODE}_${formatTimestamp(new Date())}_breathing_dataset.zip`
            triggerBlobDownload(zipBlob, zipName)
            setMessage("The ZIP package has been downloaded. You can now upload it through the form.", "info")
        } catch (error) {
            console.error(error)
            setMessage("The ZIP package could not be generated. Please try again.", "error")
        } finally {
            state.downloadInProgress = false
            refreshUi(true)
        }
    }

    function clearSession() {
        if (state.isRecording || state.isMicTestActive || state.isMicTestStopping || !state.sessionSegments.length) {
            return
        }

        const shouldClear = window.confirm("Remove all saved segments from the current session?")
        if (!shouldClear) {
            return
        }

        state.sessionSegments.forEach((segment) => revokeSegmentUrls(segment))
        state.sessionSegments = []
        state.totalSavedSeconds = 0
        state.segmentSequence = 0
        renderSavedList()
        setMessage("The current session has been cleared.", "info")
        refreshUi(true)
    }

    function handleKeydown(event) {
        const activeTag = document.activeElement ? document.activeElement.tagName : ""
        if (activeTag === "INPUT" || activeTag === "SELECT" || activeTag === "TEXTAREA") {
            return
        }

        if (event.code === "Space") {
            event.preventDefault()
            if (!state.isRecording && !state.isMicTestActive && !state.isMicTestStopping) {
                startRecording()
            }
            return
        }

        if (!state.isRecording) {
            return
        }

        const pressedKey = event.key.toLowerCase()
        if (pressedKey === "s") {
            event.preventDefault()
            stopRecording()
        } else if (pressedKey === "w") {
            event.preventDefault()
            setCurrentLabel("inhale")
        } else if (pressedKey === "e") {
            event.preventDefault()
            setCurrentLabel("exhale")
        } else if (pressedKey === "r") {
            event.preventDefault()
            setCurrentLabel("silence")
        }
    }

    function cleanupBeforeUnload() {
        state.sessionSegments.forEach((segment) => revokeSegmentUrls(segment))
        clearMicrophoneTestSample()

        if (state.mediaStream) {
            state.mediaStream.getTracks().forEach((track) => track.stop())
        }

        if (state.micTestStream) {
            state.micTestStream.getTracks().forEach((track) => track.stop())
        }

        if (state.micTestMeterFrame) {
            window.cancelAnimationFrame(state.micTestMeterFrame)
        }

        if (state.micTestSourceNode) {
            state.micTestSourceNode.disconnect()
        }

        if (state.micTestAudioContext) {
            state.micTestAudioContext.close().catch(() => undefined)
        }
    }

    dom.startButton.addEventListener("click", startRecording)
    dom.stopButton.addEventListener("click", stopRecording)
    dom.downloadZipButton.addEventListener("click", downloadZip)
    dom.clearSessionButton.addEventListener("click", clearSession)
    dom.startMicTestButton.addEventListener("click", startMicrophoneTest)
    dom.stopMicTestButton.addEventListener("click", stopMicrophoneTest)
    dom.refreshButton.addEventListener("click", async function () {
        try {
            setMessage("Refreshing the microphone list...", "info")
            await refreshMicrophones({ requestPermission: true })
            if (!state.isMicTestActive && !state.isMicTestStopping) {
                setMicrophoneTestStatus("The microphone list has been refreshed. Run the mic test if you want to verify the selected device.")
                resetMicrophoneTestMeter("Waiting for input")
            }
            setMessage("The microphone list has been refreshed.", "info")
        } catch (error) {
            console.error(error)
            setMessage("The microphone list could not be refreshed. Check browser permissions and try again.", "error")
            setMicrophoneTestStatus("The microphone list could not be refreshed. Check browser permissions and try again.")
            refreshUi(true)
        }
    })

    dom.microphoneSelect.addEventListener("change", function () {
        if (state.isRecording || state.isMicTestActive || state.isMicTestStopping) {
            return
        }

        clearMicrophoneTestSample()
        setMicrophoneTestStatus("Microphone selection updated. Run the mic test to verify the newly selected device.")
        resetMicrophoneTestMeter("Waiting for input")
        refreshUi(true)
    })

    dom.classButtons.forEach((button) => {
        button.addEventListener("click", function () {
            setCurrentLabel(button.dataset.label)
        })
    })

    if (navigator.mediaDevices && typeof navigator.mediaDevices.addEventListener === "function") {
        navigator.mediaDevices.addEventListener("devicechange", function () {
            if (!state.isRecording && !state.isMicTestActive && !state.isMicTestStopping) {
                refreshMicrophones().catch((error) => console.warn(error))
            }
        })
    }

    document.addEventListener("keydown", handleKeydown)
    window.addEventListener("beforeunload", cleanupBeforeUnload)

    renderSavedList()
    setMicrophoneTestStatus("Record a short sample, stop, and listen back before starting the full session.")
    resetMicrophoneTestMeter("Waiting for input")
    refreshUi(true)

    refreshMicrophones().catch((error) => {
        console.warn(error)
        refreshUi(true)
    })

    if (!window.JSZip) {
        setMessage("Recording will work, but ZIP export is unavailable until the page loads the JSZip library correctly.", "warning")
        refreshUi(true)
    }
})()
