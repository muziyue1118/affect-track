(() => {
    const state = {
        videos: [],
        currentIndex: 0,
        activeMode: "mock",
        socket: null,
        reconnectTimer: null,
        eegStatusTimer: null,
        lineChart: null,
        scatterChart: null,
        series: [],
    };

    const ui = {
        statusBanner: document.getElementById("demo-status"),
        statusText: document.getElementById("demo-status-text"),
        demoVideo: document.getElementById("demo-video"),
        videoSelect: document.getElementById("video-select"),
        playToggle: document.getElementById("play-toggle"),
        nextVideo: document.getElementById("next-video"),
        modeMock: document.getElementById("mode-mock"),
        modeLive: document.getElementById("mode-live"),
        startEeg: document.getElementById("start-eeg"),
        stopEeg: document.getElementById("stop-eeg"),
        activeModePill: document.getElementById("active-mode-pill"),
        streamPill: document.getElementById("stream-pill"),
        videoPill: document.getElementById("video-pill"),
        eegStatusPill: document.getElementById("eeg-status-pill"),
        valenceReadout: document.getElementById("valence-readout"),
        arousalReadout: document.getElementById("arousal-readout"),
        valenceFill: document.getElementById("valence-fill"),
        valenceKnob: document.getElementById("valence-knob"),
        arousalFill: document.getElementById("arousal-fill"),
        arousalKnob: document.getElementById("arousal-knob"),
        lineChart: document.getElementById("line-chart"),
        scatterChart: document.getElementById("scatter-chart"),
    };

    function setStatus(message, tone = "info") {
        ui.statusBanner.dataset.tone = tone;
        ui.statusText.textContent = message;
    }

    function setActiveMode(mode) {
        state.activeMode = mode;
        ui.modeMock.classList.toggle("is-active", mode === "mock");
        ui.modeLive.classList.toggle("is-active", mode === "live");
        ui.activeModePill.textContent = `当前模式：${mode}`;
    }

    async function fetchJson(url, options) {
        const response = await fetch(url, options);
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || "请求失败");
        }
        return response.json();
    }

    async function loadVideos() {
        state.videos = await fetchJson("/api/videos");
        ui.videoSelect.innerHTML = "";
        if (!state.videos.length) {
            const option = document.createElement("option");
            option.textContent = "暂无可用视频";
            option.value = "";
            ui.videoSelect.appendChild(option);
            ui.videoSelect.disabled = true;
            ui.playToggle.disabled = true;
            ui.nextVideo.disabled = true;
            ui.videoPill.textContent = "当前视频：无";
            setStatus("没有检测到合规视频，请先按 positive_1.mp4 规则重命名素材。", "warning");
            return;
        }

        state.videos.forEach((video, index) => {
            const option = document.createElement("option");
            option.value = String(index);
            option.textContent = `${video.category.toUpperCase()} · ${video.name}`;
            ui.videoSelect.appendChild(option);
        });

        ui.videoSelect.disabled = false;
        ui.playToggle.disabled = false;
        ui.nextVideo.disabled = false;
        await playVideoAt(0);
    }

    async function playVideoAt(index) {
        if (!state.videos.length) {
            return;
        }
        state.currentIndex = (index + state.videos.length) % state.videos.length;
        const current = state.videos[state.currentIndex];
        ui.videoSelect.value = String(state.currentIndex);
        ui.videoPill.textContent = `当前视频：${current.name}`;
        ui.demoVideo.pause();
        ui.demoVideo.src = current.url;
        ui.demoVideo.load();
        try {
            await ui.demoVideo.play();
            ui.playToggle.textContent = "暂停";
            setStatus(`正在播放在线演示视频 ${current.name}。`, "info");
        } catch (_error) {
            ui.playToggle.textContent = "播放";
            setStatus("浏览器阻止了自动播放，请手动点击播放。", "warning");
        }
    }

    function initCharts() {
        if (!window.echarts) {
            ui.lineChart.innerHTML = '<div class="empty-state">本地 ECharts 资源未就绪，请确认 /static/vendor/echarts.min.js 已存在。</div>';
            ui.scatterChart.innerHTML = '<div class="empty-state">等待 ECharts 本地脚本加载。</div>';
            return;
        }

        state.lineChart = window.echarts.init(ui.lineChart);
        state.scatterChart = window.echarts.init(ui.scatterChart);

        state.lineChart.setOption({
            backgroundColor: "transparent",
            animationDurationUpdate: 160,
            tooltip: {
                trigger: "axis",
                valueFormatter: (value) => Number(value).toFixed(2),
            },
            legend: {
                top: 6,
                data: ["Valence", "Arousal"],
            },
            grid: {
                top: 48,
                right: 18,
                bottom: 28,
                left: 42,
            },
            xAxis: {
                type: "time",
                boundaryGap: false,
                axisLabel: {
                    formatter: (value) => new Date(value).toLocaleTimeString("zh-CN", { hour12: false }),
                },
            },
            yAxis: {
                type: "value",
                min: 1,
                max: 5,
                splitNumber: 4,
            },
            series: [
                {
                    name: "Valence",
                    type: "line",
                    smooth: true,
                    showSymbol: false,
                    lineStyle: { width: 3, color: "#2a9d8f" },
                    areaStyle: { color: "rgba(42, 157, 143, 0.14)" },
                    data: [],
                },
                {
                    name: "Arousal",
                    type: "line",
                    smooth: true,
                    showSymbol: false,
                    lineStyle: { width: 3, color: "#ef6f55" },
                    areaStyle: { color: "rgba(239, 111, 85, 0.12)" },
                    data: [],
                },
            ],
        });

        state.scatterChart.setOption({
            backgroundColor: "transparent",
            animationDurationUpdate: 180,
            grid: {
                top: 24,
                right: 24,
                bottom: 42,
                left: 46,
            },
            xAxis: {
                type: "value",
                name: "Valence",
                min: 1,
                max: 5,
                splitNumber: 4,
            },
            yAxis: {
                type: "value",
                name: "Arousal",
                min: 1,
                max: 5,
                splitNumber: 4,
            },
            series: [
                {
                    type: "effectScatter",
                    rippleEffect: { scale: 3.5, brushType: "stroke" },
                    symbolSize: 20,
                    itemStyle: {
                        color: "#2f9bff",
                        shadowBlur: 18,
                        shadowColor: "rgba(47, 155, 255, 0.45)",
                    },
                    data: [[3, 3]],
                },
            ],
        });

        window.addEventListener("resize", () => {
            state.lineChart?.resize();
            state.scatterChart?.resize();
        });
    }

    function updateCharts(frame) {
        if (!state.lineChart || !state.scatterChart) {
            return;
        }

        const now = Date.now();
        state.series.push({
            time: now,
            valence: Number(frame.valence),
            arousal: Number(frame.arousal),
        });
        const cutoff = now - 30000;
        state.series = state.series.filter((item) => item.time >= cutoff);

        state.lineChart.setOption({
            xAxis: {
                min: cutoff,
                max: now,
            },
            series: [
                {
                    data: state.series.map((item) => [item.time, item.valence]),
                },
                {
                    data: state.series.map((item) => [item.time, item.arousal]),
                },
            ],
        });

        state.scatterChart.setOption({
            series: [
                {
                    data: [[Number(frame.valence), Number(frame.arousal)]],
                },
            ],
        });
    }

    function setMeter(kind, value) {
        const numeric = Math.min(5, Math.max(1, Number(value)));
        const percent = ((numeric - 1) / 4) * 100;
        const fill = kind === "valence" ? ui.valenceFill : ui.arousalFill;
        const knob = kind === "valence" ? ui.valenceKnob : ui.arousalKnob;
        const root = fill?.closest(".emotion-meter");
        root?.style.setProperty("--meter-percent", `${percent}%`);
        if (fill) {
            fill.style.width = `${percent}%`;
        }
        if (knob) {
            knob.style.left = `${percent}%`;
        }
    }

    function updateReadout(frame) {
        ui.valenceReadout.textContent = Number(frame.valence).toFixed(2);
        ui.arousalReadout.textContent = Number(frame.arousal).toFixed(2);
        setMeter("valence", frame.valence);
        setMeter("arousal", frame.arousal);
        ui.streamPill.textContent = `WebSocket 已连接 · ${frame.source || state.activeMode}`;
        updateCharts(frame);
    }

    function connectSocket() {
        if (state.socket) {
            state.socket.close();
        }

        const protocol = window.location.protocol === "https:" ? "wss" : "ws";
        const socket = new WebSocket(`${protocol}://${window.location.host}/ws/emotion_stream`);
        state.socket = socket;

        socket.addEventListener("open", () => {
            ui.streamPill.textContent = "WebSocket 已连接";
            setStatus("在线情绪流已连接，正在等待数据。", "success");
        });

        socket.addEventListener("message", (event) => {
            try {
                const frame = JSON.parse(event.data);
                updateReadout(frame);
            } catch (_error) {
                setStatus("收到无法解析的实时数据。", "warning");
            }
        });

        socket.addEventListener("close", () => {
            ui.streamPill.textContent = "WebSocket 已断开";
            setStatus("实时连接已断开，正在尝试自动重连。", "warning");
            clearTimeout(state.reconnectTimer);
            state.reconnectTimer = window.setTimeout(connectSocket, 1500);
        });

        socket.addEventListener("error", () => {
            setStatus("WebSocket 连接出现异常。", "error");
        });
    }

    async function syncMode() {
        const payload = await fetchJson("/api/emotion_mode");
        setActiveMode(payload.mode);
    }

    async function setMode(mode) {
        const payload = await fetchJson("/api/emotion_mode", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ mode }),
        });
        setActiveMode(payload.mode);
        setStatus(payload.mode === "mock" ? "已切换到模拟情绪流。" : "已切换到 live 模式，等待模型推流。", "success");
    }

    function renderEegStatus(status) {
        const prefix = status.running ? "EEG：运行中" : status.model_ready ? "EEG：模型就绪" : "EEG：未就绪";
        ui.eegStatusPill.textContent = `${prefix} · ${status.status || "unknown"}`;
        ui.startEeg.disabled = Boolean(status.running);
        ui.stopEeg.disabled = !status.running;
    }

    async function refreshEegStatus() {
        try {
            const status = await fetchJson("/api/online_eeg/status");
            renderEegStatus(status);
        } catch (_error) {
            ui.eegStatusPill.textContent = "EEG：状态不可用";
        }
    }

    async function startOnlineEeg() {
        const status = await fetchJson("/api/online_eeg/start", { method: "POST" });
        renderEegStatus(status);
        await syncMode();
        setStatus(status.running ? "EEG 在线推理已启动，正在等待有效窗口。" : status.message, status.running ? "success" : "warning");
    }

    async function stopOnlineEeg() {
        const status = await fetchJson("/api/online_eeg/stop", { method: "POST" });
        renderEegStatus(status);
        setStatus("EEG 在线推理已停止。", "info");
    }

    ui.modeMock.addEventListener("click", () => setMode("mock"));
    ui.modeLive.addEventListener("click", () => setMode("live"));
    ui.startEeg.addEventListener("click", () => startOnlineEeg().catch((error) => setStatus(error.message, "error")));
    ui.stopEeg.addEventListener("click", () => stopOnlineEeg().catch((error) => setStatus(error.message, "error")));
    ui.videoSelect.addEventListener("change", (event) => playVideoAt(Number(event.target.value)));
    ui.playToggle.addEventListener("click", async () => {
        if (ui.demoVideo.paused) {
            await ui.demoVideo.play();
            ui.playToggle.textContent = "暂停";
        } else {
            ui.demoVideo.pause();
            ui.playToggle.textContent = "播放";
        }
    });
    ui.nextVideo.addEventListener("click", () => playVideoAt(state.currentIndex + 1));
    ui.demoVideo.addEventListener("ended", () => playVideoAt(state.currentIndex + 1));

    (async () => {
        initCharts();
        try {
            await Promise.all([loadVideos(), syncMode(), refreshEegStatus()]);
            connectSocket();
            state.eegStatusTimer = window.setInterval(refreshEegStatus, 2500);
            setStatus("在线演示页面已就绪。", "success");
        } catch (error) {
            setStatus(error.message || "在线演示初始化失败。", "error");
        }
    })();
})();
