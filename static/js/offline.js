(() => {
    const state = {
        subjectId: "",
        videos: [],
        queue: [],
        currentIndex: -1,
        currentVideo: null,
        currentStartTime: "",
        isSaving: false,
        touched: {
            valence: false,
            arousal: false,
        },
    };

    const PLAYBACK_KEYS = new Set([" ", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "k", "K"]);

    const ui = {
        statusBanner: document.getElementById("status-banner"),
        statusText: document.getElementById("status-text"),
        loginState: document.getElementById("state-login"),
        playbackState: document.getElementById("state-playback"),
        scoringState: document.getElementById("state-scoring"),
        completeState: document.getElementById("state-complete"),
        subjectForm: document.getElementById("subject-form"),
        subjectIdInput: document.getElementById("subject-id"),
        startButton: document.getElementById("start-button"),
        experimentVideo: document.getElementById("experiment-video"),
        manualPlayButton: document.getElementById("manual-play-button"),
        metaSubject: document.getElementById("meta-subject"),
        metaVideo: document.getElementById("meta-video"),
        metaStartTime: document.getElementById("meta-start-time"),
        scoreVideoCaption: document.getElementById("score-video-caption"),
        progressPill: document.getElementById("progress-pill"),
        valenceRange: document.getElementById("valence-range"),
        arousalRange: document.getElementById("arousal-range"),
        valenceValue: document.getElementById("valence-value"),
        arousalValue: document.getElementById("arousal-value"),
        submitButton: document.getElementById("submit-score-button"),
        scoreProgressCopy: document.getElementById("score-progress-copy"),
        completeCount: document.getElementById("complete-count"),
        completeSubject: document.getElementById("complete-subject"),
    };

    function setStatus(message, tone = "info") {
        ui.statusBanner.dataset.tone = tone;
        ui.statusText.textContent = message;
    }

    function activateState(target) {
        [ui.loginState, ui.playbackState, ui.scoringState, ui.completeState].forEach((section) => {
            section.classList.toggle("is-active", section === target);
        });
    }

    function pad(value, size = 2) {
        return String(value).padStart(size, "0");
    }

    function generateExperimentTimestamp() {
        const now = new Date();
        const datePart = [
            now.getFullYear(),
            pad(now.getMonth() + 1),
            pad(now.getDate()),
            pad(now.getHours()),
            pad(now.getMinutes()),
            pad(now.getSeconds()),
        ].join("");
        const milliseconds = pad(now.getMilliseconds(), 3);
        const extraDigit = Math.floor((performance.now() % 1) * 10);
        return `E${datePart}${milliseconds}${extraDigit}`;
    }

    function shuffleVideos(videos) {
        const queue = [...videos];
        for (let i = queue.length - 1; i > 0; i -= 1) {
            const j = Math.floor(Math.random() * (i + 1));
            [queue[i], queue[j]] = [queue[j], queue[i]];
        }
        return queue;
    }

    async function fetchVideos() {
        const response = await fetch("/api/videos");
        if (!response.ok) {
            throw new Error("无法加载视频列表。");
        }
        return response.json();
    }

    async function requestFullscreen() {
        if (!document.fullscreenElement && document.documentElement.requestFullscreen) {
            try {
                await document.documentElement.requestFullscreen();
            } catch (_error) {
                setStatus("浏览器拒绝进入全屏，实验仍可继续进行。", "warning");
            }
        }
    }

    async function exitFullscreen() {
        if (document.fullscreenElement && document.exitFullscreen) {
            try {
                await document.exitFullscreen();
            } catch (_error) {
                // Ignore exit failures to avoid blocking completion.
            }
        }
    }

    function resetScoringControls() {
        state.touched.valence = false;
        state.touched.arousal = false;
        ui.valenceRange.value = "3";
        ui.arousalRange.value = "3";
        ui.valenceValue.textContent = "未选择";
        ui.arousalValue.textContent = "未选择";
        ui.valenceValue.classList.add("is-pending");
        ui.arousalValue.classList.add("is-pending");
        ui.submitButton.disabled = true;
        ui.submitButton.textContent = "提交并继续";
    }

    function updateScoreAvailability() {
        const ready = state.touched.valence && state.touched.arousal && !state.isSaving;
        ui.submitButton.disabled = !ready;
        ui.progressPill.textContent = ready ? "可提交" : `第 ${state.currentIndex + 1} / ${state.queue.length} 段`;
        ui.progressPill.classList.toggle("is-pending", !ready);
        ui.scoreProgressCopy.textContent = ready
            ? "评分就绪，提交后会立即写入 CSV 并进入下一段视频。"
            : "请先完成两个维度的评分，按钮才会解锁。";
    }

    function updateRangeValue(kind, value) {
        const target = kind === "valence" ? ui.valenceValue : ui.arousalValue;
        target.textContent = `${value} 分`;
        target.classList.remove("is-pending");
        state.touched[kind] = true;
        updateScoreAvailability();
    }

    async function attemptPlayback() {
        ui.manualPlayButton.hidden = true;
        try {
            await ui.experimentVideo.play();
        } catch (_error) {
            ui.manualPlayButton.hidden = false;
            setStatus("浏览器拦截了自动播放，请点击“点击开始播放”。", "warning");
        }
    }

    async function startNextVideo() {
        state.currentIndex += 1;
        if (state.currentIndex >= state.queue.length) {
            await finishExperiment();
            return;
        }

        state.currentVideo = state.queue[state.currentIndex];
        state.currentStartTime = "";
        resetScoringControls();
        activateState(ui.playbackState);

        ui.metaSubject.textContent = state.subjectId;
        ui.metaVideo.textContent = state.currentVideo.name;
        ui.metaStartTime.textContent = "等待播放";
        ui.scoreVideoCaption.textContent = `当前视频：${state.currentVideo.name}`;
        ui.progressPill.textContent = `第 ${state.currentIndex + 1} / ${state.queue.length} 段`;
        setStatus(`正在准备播放第 ${state.currentIndex + 1} 段视频。`, "info");

        ui.experimentVideo.pause();
        ui.experimentVideo.removeAttribute("src");
        ui.experimentVideo.load();
        ui.experimentVideo.src = state.currentVideo.url;
        ui.experimentVideo.load();
        await attemptPlayback();
    }

    async function finishExperiment() {
        await exitFullscreen();
        activateState(ui.completeState);
        ui.completeCount.textContent = String(state.queue.length);
        ui.completeSubject.textContent = state.subjectId;
        setStatus("实验已完成，页面已退出全屏。请实验员确认记录结果。", "success");
    }

    async function handleStart(event) {
        event.preventDefault();
        const subjectId = ui.subjectIdInput.value.trim();
        if (!subjectId) {
            setStatus("请先输入受试者编号。", "error");
            return;
        }

        state.subjectId = subjectId;
        state.currentIndex = -1;
        state.currentVideo = null;
        state.isSaving = false;
        ui.startButton.disabled = true;
        setStatus("正在加载视频列表并生成随机播放队列。", "info");

        try {
            state.videos = await fetchVideos();
        } catch (error) {
            ui.startButton.disabled = false;
            setStatus(error.message || "加载视频列表失败。", "error");
            return;
        }

        if (!state.videos.length) {
            ui.startButton.disabled = false;
            setStatus("没有检测到合法视频。请先将 video 目录中的文件重命名为 positive_1.mp4 等规范格式。", "error");
            return;
        }

        state.queue = shuffleVideos(state.videos);
        await requestFullscreen();
        await startNextVideo();
    }

    function handleVideoPlaying() {
        if (!state.currentStartTime) {
            state.currentStartTime = generateExperimentTimestamp();
            ui.metaStartTime.textContent = state.currentStartTime;
            setStatus(`视频 ${state.currentVideo.name} 已开始播放。`, "info");
        }
    }

    function handleVideoEnded() {
        activateState(ui.scoringState);
        setStatus(`视频 ${state.currentVideo.name} 播放结束，请完成评分。`, "info");
    }

    async function submitScore() {
        if (!state.currentVideo || !state.currentStartTime || state.isSaving) {
            return;
        }

        if (!state.touched.valence || !state.touched.arousal) {
            setStatus("请先完成两个维度的评分。", "warning");
            return;
        }

        state.isSaving = true;
        updateScoreAvailability();
        ui.submitButton.textContent = "保存中...";

        const payload = {
            subject_id: state.subjectId,
            video_name: state.currentVideo.name,
            start_time: state.currentStartTime,
            valence: Number(ui.valenceRange.value),
            arousal: Number(ui.arousalRange.value),
        };

        try {
            const response = await fetch("/api/save_score", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                const errorBody = await response.json().catch(() => ({}));
                throw new Error(errorBody.detail || "保存失败，请重试。");
            }

            setStatus(`已保存 ${state.currentVideo.name} 的评分记录。`, "success");
            state.isSaving = false;
            await startNextVideo();
        } catch (error) {
            state.isSaving = false;
            ui.submitButton.textContent = "提交并继续";
            updateScoreAvailability();
            setStatus(error.message || "保存失败，请重试。", "error");
        }
    }

    function handleBeforeUnload(event) {
        const activeExperiment = state.subjectId && state.currentIndex >= 0 && state.currentIndex < state.queue.length;
        if (activeExperiment) {
            event.preventDefault();
            event.returnValue = "";
        }
    }

    function handlePlaybackKeys(event) {
        if (!ui.playbackState.classList.contains("is-active")) {
            return;
        }
        if (PLAYBACK_KEYS.has(event.key)) {
            event.preventDefault();
        }
    }

    ui.subjectForm.addEventListener("submit", handleStart);
    ui.manualPlayButton.addEventListener("click", attemptPlayback);
    ui.submitButton.addEventListener("click", submitScore);
    ui.valenceRange.addEventListener("input", (event) => updateRangeValue("valence", event.target.value));
    ui.arousalRange.addEventListener("input", (event) => updateRangeValue("arousal", event.target.value));
    ui.experimentVideo.addEventListener("playing", handleVideoPlaying);
    ui.experimentVideo.addEventListener("ended", handleVideoEnded);
    document.addEventListener("keydown", handlePlaybackKeys, { passive: false });
    window.addEventListener("beforeunload", handleBeforeUnload);
})();
