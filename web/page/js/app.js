// 掌柜智库一体化前端逻辑

// 1. 初始化变量与 API 基地址
const API_BASE = window.location.origin.startsWith('http') ? window.location.origin : 'http://127.0.0.1:8000';
let sessionId = localStorage.getItem('kb_session_id');
if (!sessionId) {
    sessionId = 'sess-' + Math.random().toString(36).slice(2) + Date.now().toString(36);
    localStorage.setItem('kb_session_id', sessionId);
}
let currentSessionChunks = {}; // 全局缓存当前提问检索到的切片数据

// 获取常用 DOM 元素
const apiPill = document.getElementById('apiPill');
const navItems = document.querySelectorAll('.nav-item');
const tabPanes = document.querySelectorAll('.tab-pane');

// 2. 侧边栏 Tab 切换逻辑
navItems.forEach(item => {
    item.addEventListener('click', () => {
        // 移除所有的 active 状态
        navItems.forEach(nav => nav.classList.remove('active'));
        tabPanes.forEach(pane => pane.classList.remove('active'));

        // 激活当前点击的 Tab
        item.classList.add('active');
        const targetTab = item.getAttribute('data-tab');
        document.getElementById(targetTab).classList.add('active');

        // 切换到问答界面时，自动聚焦输入框
        if (targetTab === 'tab-chat') {
            setTimeout(() => inputEl.focus(), 100);
        }
    });
});

// 3. API 健康度检查
async function apiHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        if (!res.ok) throw new Error('Health check response not OK');
        apiPill.textContent = 'API: 已连接';
        apiPill.style.color = '#16a34a'; // 高清晰度健康绿色字体
        apiPill.style.borderColor = '#cbd5e1';
        apiPill.style.borderBottomColor = '#16a34a'; // 下边框健康绿
    } catch (e) {
        apiPill.textContent = 'API: 未连接';
        apiPill.style.color = '#dc2626'; // 警告红色字体
        apiPill.style.borderColor = '#cbd5e1';
        apiPill.style.borderBottomColor = '#dc2626'; // 下边框警告红
    }
}

// ---------------- 智能问答模块 (Chat) ----------------
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const btnClear = document.getElementById('btnClear');

function scrollToBottom() {
    chatEl.scrollTop = chatEl.scrollHeight;
}

function nowTime() {
    const d = new Date();
    return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

function formatTime(ts) {
    if (!ts) return nowTime();
    const d = new Date(Number(ts) * 1000);
    if (Number.isNaN(d.getTime())) return nowTime();
    return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

function escapeHtml(str) {
    return String(str)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", "&#039;");
}

function isImageUrl(url) {
    try {
        const u = new URL(url);
        return /\.(png|jpe?g|gif|webp|bmp|svg)$/i.test(u.pathname);
    } catch (e) {
        return /\.(png|jpe?g|gif|webp|bmp|svg)(\?|#|$)/i.test(url || '');
    }
}

function normalizeUrl(rawUrl) {
    const s = String(rawUrl || '').trim();
    if (!s) return '';
    return s.replace(/\s/g, '%20');
}

window.sendConfirmOption = function(btnEl, optionText) {
    // 禁用当前选项卡下的所有按钮防止多次触发
    const parent = btnEl.closest('.confirm-buttons-grid');
    if (parent) {
        const buttons = parent.querySelectorAll('.btn-confirm-opt');
        buttons.forEach(btn => {
            btn.disabled = true;
            btn.style.opacity = '0.6';
            btn.style.cursor = 'not-allowed';
        });
    }
    // 填入输入框并触发发送
    const inputEl = document.getElementById('input');
    if (inputEl) {
        inputEl.value = optionText;
        const sendBtn = document.getElementById('send');
        if (sendBtn) {
            sendBtn.click();
        }
    }
};

function parseConfirmOptions(text) {
    if (text && text.includes("您想咨询的是以下哪一个")) {
        const trigger = "您想咨询的是以下哪一个?";
        const idx = text.indexOf(trigger);
        const promptText = text.substring(0, idx + trigger.length);
        let optionsPart = text.substring(idx + trigger.length);
        
        // 清洗前导的换行符、冒号等字符
        optionsPart = optionsPart.replace(/^[\s:\n：\r]+/g, '');
        
        // 根据制表符、换行、连续的两个及以上空格或逗号切割选项
        const rawOptions = optionsPart.split(/[\t\n,，\r]|\s{2,}/);
        const options = rawOptions
            .map(opt => opt.trim())
            .filter(opt => opt && opt !== "空字符串" && opt !== "None" && opt !== "null");
            
        if (options.length > 0) {
            return {
                prompt: promptText,
                options: options
            };
        }
    }
    return null;
}

// 引用角标重编号：把答案中的 [[chunk_id]] 按首次出现顺序重编号为 [1][2]...（同一 chunk 复用同一编号，点击仍按 chunk_id 打开原文抽屉）
function linkCitations(mdText) {
    const seqByChunkId = new Map();
    return mdText.replace(/\[\[(\d+)\]\]/g, (match, chunkId) => {
        if (!seqByChunkId.has(chunkId)) {
            seqByChunkId.set(chunkId, seqByChunkId.size + 1);
        }
        return `<span class="citation-badge" onclick="openSourceDrawer('${chunkId}')">[${seqByChunkId.get(chunkId)}]</span>`;
    });
}

// 原文（chunk）Markdown 渲染：图片/链接 URL 空格转义后交给 marked，渲染为 HTML 供引用抽屉展示（图片将以 <img> 真实加载）
function formatChunkToHtml(chunkText) {
    if (!chunkText) return '';
    let mdText = String(chunkText);
    mdText = mdText.replace(/!\[(.*?)\]\((.*?)\)/g, (match, alt, url) => {
        const cleanUrl = url.trim().replace(/\s+/g, '%20');
        return `![${alt}](${cleanUrl})`;
    });
    mdText = mdText.replace(/(?<!\!)\[(.*?)\]\((.*?)\)/g, (match, text, url) => {
        const cleanUrl = url.trim().replace(/\s+/g, '%20');
        return `[${text}](${cleanUrl})`;
    });
    if (typeof marked !== 'undefined' && typeof marked.parse === 'function') {
        return marked.parse(mdText);
    }
    return escapeHtml(mdText).replace(/\r?\n/g, '<br>');
}

function formatAnswerToHtml(answerText) {
    if (!answerText) return '';

    // 检测是否为中等置信度确认列表，是则直接渲染为交互按钮卡片
    const optionData = parseConfirmOptions(answerText);
    if (optionData) {
        let html = `<div class="confirm-options-wrap">`;
        html += `<p class="confirm-prompt">${escapeHtml(optionData.prompt)}</p>`;
        html += `<div class="confirm-buttons-grid">`;
        optionData.options.forEach(opt => {
            html += `<button class="btn-confirm-opt" onclick="sendConfirmOption(this, '${escapeHtml(opt)}')">${escapeHtml(opt)}</button>`;
        });
        html += `</div></div>`;
        return html;
    }
    
    let mdText = answerText;
 
    // 0. 预处理：识别可能包含空格但以常见图片扩展名结尾的 URL，将其中的空格替换为 %20，防止链接在空格处截断
    const rawImgWithSpaceRegex = /(https?:\/\/[^\s<>\u007f-\u009f]+?(?:\s+[^\s<>\u007f-\u009f]+)*?\.(?:png|jpe?g|gif|webp|bmp|svg)(?:\?[^\s<>#]*)?)/gi;
    mdText = mdText.replace(rawImgWithSpaceRegex, (match) => {
        return match.replace(/\s+/g, '%20');
    });

    // 0.5 预处理：识别普通链接格式中其实是指向图片的链接 [text](image_url)，自动转换为 Markdown 图片语法 ![text](image_url) 格式直接渲染图片
    const imgLinkRegex = /(?<!\!)\[(.*?)\]\((https?:\/\/[^)\u007f-\u009f]+?\.(?:png|jpe?g|gif|webp|bmp|svg)(?:\?[^)#]*)?)\)/gi;
    mdText = mdText.replace(imgLinkRegex, (match, text, url) => {
        return `![${text}](${url})`;
    });

    // 1. 预处理：匹配 Markdown 图片，自动将其 URL 中的空格转义为 %20 防止解析失败
    mdText = mdText.replace(/\!\[(.*?)\]\((.*?)\)/g, (match, alt, url) => {
        const cleanUrl = url.trim().replace(/\s+/g, '%20');
        return `![${alt}](${cleanUrl})`;
    });

    // 2. 预处理：匹配 Markdown 链接，自动将其 URL 中的空格转义为 %20
    mdText = mdText.replace(/(?<!\!)\[(.*?)\]\((.*?)\)/g, (match, text, url) => {
        const cleanUrl = url.trim().replace(/\s+/g, '%20');
        return `[${text}](${cleanUrl})`;
    });

    // 3. 匹配未被 Markdown 图片或链接格式包裹的裸图片 URL，并自动包装为 ![图片](url) 格式
    const imgRegex = /(?<!\()https?:\/\/[^\s<>\u007f-\u009f]+?\.(?:png|jpe?g|gif|webp|bmp|svg)(?:\?[^\s<>#]*)?/gi;
    mdText = mdText.replace(imgRegex, (url) => `![图片](${url})`);

    // 3.5 识别引用标注 [[chunk_id]] 格式并替换为可点击的引用角标（按出现顺序重编号，见 linkCitations）
    mdText = linkCitations(mdText);

    // 调用 marked 解析库解析 Markdown，如果未引入则 fallback 降级为普通文本
    let html = '';
    if (typeof marked !== 'undefined' && typeof marked.parse === 'function') {
        html = marked.parse(mdText);
    } else {
        html = escapeHtml(mdText).replace(/\r?\n/g, '<br>');
    }

    return html;
}

function renderAnswerWithImages(containerEl, answerText, candidateImageUrls) {
    let htmlContent = formatAnswerToHtml(answerText || '');

    const candidates = Array.isArray(candidateImageUrls)
        ? candidateImageUrls.map(normalizeUrl).filter(isImageUrl)
        : [];

    if (candidates.length > 0) {
        let additionalHtml = '';
        for (const url of candidates) {
            if (!answerText.includes(url)) {
                additionalHtml += `<div class="answer-img-wrap"><img src="${url}" loading="lazy" alt="参考图片" referrerPolicy="no-referrer" onerror="this.parentNode.style.display='none'"></div>`;
            }
        }
        if (additionalHtml) {
            htmlContent += `<div class="answer-images" style="margin-top: 10px;">${additionalHtml}</div>`;
        }
    }

    containerEl.innerHTML = htmlContent.trim() || '（已完成，但未返回答案）';
}

function addUserMsg(text) {
    const html = `
    <div class="msg user">
      <div>
        <div class="bubble">${escapeHtml(text)}</div>
        <div class="meta">${nowTime()}</div>
      </div>
      <div class="avatar">我</div>
    </div>
  `;
    chatEl.insertAdjacentHTML('beforeend', html);
    scrollToBottom();
}

function addUserMsgWithTime(text, ts) {
    const html = `
    <div class="msg user">
      <div>
        <div class="bubble">${escapeHtml(text)}</div>
        <div class="meta">${formatTime(ts)}</div>
      </div>
      <div class="avatar">我</div>
    </div>
  `;
    chatEl.insertAdjacentHTML('beforeend', html);
    scrollToBottom();
}

function addBotMsgWithTime(text, ts, imageUrls) {
    const id = 'bot-his-' + Math.random().toString(36).slice(2);
    const html = `
    <div class="msg bot" id="${id}">
      <div class="avatar bot">🤖</div>
      <div>
        <div class="bubble"><div class="answer"></div></div>
        <div class="meta">${formatTime(ts)}</div>
      </div>
    </div>
  `;
    chatEl.insertAdjacentHTML('beforeend', html);
    const el = document.getElementById(id);
    if (el) {
        const answerEl = el.querySelector('.answer');
        renderAnswerWithImages(answerEl, text || '', imageUrls || []);
    }
    scrollToBottom();
}

function addBotMsgSkeleton() {
    const id = 'bot-' + Math.random().toString(36).slice(2);
    const html = `
    <div class="msg bot" id="${id}">
      <div class="avatar bot">🤖</div>
      <div style="min-width: 180px;">
        <div class="bubble">
          <span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>
          <details class="progress" open>
            <summary>阶段进度（等待中）</summary>
            <ul></ul>
          </details>
        </div>
        <div class="meta">${nowTime()}</div>
      </div>
    </div>
  `;
    chatEl.insertAdjacentHTML('beforeend', html);
    scrollToBottom();
    return document.getElementById(id);
}

function renderProgress(botMsgEl, doneList, runningList, status) {
    const details = botMsgEl.querySelector('details.progress');
    if (!details) return;
    const summary = details.querySelector('summary');
    const ul = details.querySelector('ul');
    const done = Array.isArray(doneList) ? doneList : [];
    const running = Array.isArray(runningList) ? runningList : [];
    const totalDone = done.length;
    const totalRun = running.length;

    const statusMap = {
        'processing': '处理中',
        'completed': '已完成',
        'failed': '失败',
        'pending': '等待中',
    };
    const displayStatus = statusMap[status] || status || 'unknown';

    summary.textContent = `阶段进度（已完成 ${totalDone}，进行中 ${totalRun}，状态：${displayStatus}）`;

    const lines = [
        ...done.map(x => `✅ ${x}`),
        ...running.map(x => `⏳ ${x}`)
    ];
    ul.innerHTML = '';
    if (lines.length === 0) {
        ul.insertAdjacentHTML('beforeend', '<li>暂无进度</li>');
    } else {
        for (const line of lines) {
            ul.insertAdjacentHTML('beforeend', `<li>${escapeHtml(line)}</li>`);
        }
    }
}

async function loadHistory() {
    try {
        console.log("正在请求会话历史，会话ID:", sessionId);
        const res = await fetch(`${API_BASE}/history/${sessionId}`);
        if (!res.ok) {
            console.error("请求历史记录接口失败，状态码:", res.status);
            return;
        }
        const data = await res.json();
        console.log("成功获取到历史记录数据:", data);
        const items = Array.isArray(data.items) ? data.items : [];
        
        // 保留首条欢迎消息，其余先清空再渲染历史
        const nodes = Array.from(chatEl.querySelectorAll('.msg'));
        for (let i = 1; i < nodes.length; i++) nodes[i].remove();
        
        for (const item of items) {
            if (item.role === 'user') {
                addUserMsgWithTime(item.text || '', item.ts);
            } else {
                if (Array.isArray(item.chunks)) {
                    item.chunks.forEach(c => {
                        currentSessionChunks[c.id] = c;
                    });
                }
                addBotMsgWithTime(item.text || '', item.ts, item.image_urls || []);
            }
        }
        scrollToBottom();
    } catch (e) {
        console.error("解析或渲染历史记录时出错:", e);
    }
}

async function submitQuery(text) {
    const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            query: text,
            session_id: sessionId
        })
    });
    if (!res.ok) {
        const msg = await res.text();
        throw new Error(msg || '请求失败');
    }
    return await res.json();
}

async function onSend() {
    const text = (inputEl.value || '').trim();
    if (!text) return;
    currentSessionChunks = {}; // 新提问开始，清空上一轮的缓存切片，防止内存泄露和交叉干扰
    inputEl.value = '';
    addUserMsg(text);
    const botMsgEl = addBotMsgSkeleton();
    sendBtn.disabled = true;

    try {
        const data = await submitQuery(text);
        renderProgress(botMsgEl, [], [], 'pending');

        const { task_id } = data;
        const bubble = botMsgEl.querySelector('.bubble');
        let answerEl = bubble.querySelector('.answer');
        if (!answerEl) {
            answerEl = document.createElement('div');
            answerEl.className = 'answer';
            bubble.insertBefore(answerEl, bubble.firstChild);
        }

        const es = new EventSource(`${API_BASE}/stream/${task_id}`);
        let rawAnswerText = '';

        es.addEventListener('chunks', (e) => {
            try {
                const chunks = JSON.parse(e.data || '[]');
                chunks.forEach(c => {
                    currentSessionChunks[c.id] = c;
                });
            } catch (_) {}
        });

        es.addEventListener('progress', (e) => {
            try {
                const d = JSON.parse(e.data || '{}');
                renderProgress(botMsgEl, d.done_list, d.running_list, d.status);
                if (d && d.status === 'completed') {
                    const typing = botMsgEl.querySelector('.typing');
                    if (typing) typing.remove();
                    sendBtn.disabled = false;
                }
            } catch (_) {}
        });

        es.addEventListener('delta', (e) => {
            try {
                const d = JSON.parse(e.data || '{}');
                const delta = d.delta || '';
                if (delta) {
                    rawAnswerText += delta;
                    renderAnswerWithImages(answerEl, rawAnswerText, []);
                    scrollToBottom();
                }
            } catch (_) {}
        });

        es.addEventListener('final', (e) => {
            const typing = botMsgEl.querySelector('.typing');
            if (typing) typing.remove();

            // 强制把所有"进行中 ⏳"的节点标记为"已完成 ✅"
            const details = botMsgEl.querySelector('details.progress');
            if (details) {
                const ul = details.querySelector('ul');
                if (ul) {
                    const lis = ul.querySelectorAll('li');
                    lis.forEach(li => {
                        if (li.textContent.includes('⏳')) {
                            li.textContent = li.textContent.replace('⏳', '✅');
                        }
                    });
                }
                const summary = details.querySelector('summary');
                if (summary) {
                    const totalDone = details.querySelectorAll('li').length;
                    summary.textContent = `阶段进度（已完成 ${totalDone}，进行中 0，状态：已完成）`;
                }
                details.removeAttribute('open');
            }

            try {
                const d = JSON.parse(e.data || '{}');
                const finalText = (d && typeof d.answer === 'string' && d.answer.trim().length > 0) ? d.answer : (rawAnswerText || answerEl.textContent || '');
                renderAnswerWithImages(answerEl, finalText, d.image_urls || []);
            } catch (_) {}
            
            es.close();
            sendBtn.disabled = false;
            scrollToBottom();
        });

        es.addEventListener('error', (e) => {
            const typing = botMsgEl.querySelector('.typing');
            if (typing) typing.remove();
            
            const progress = botMsgEl.querySelector('details.progress');
            if (progress) progress.removeAttribute('open');

            try {
                const msg = e && e.data ? (JSON.parse(e.data).error || 'SSE 连接中断/失败') : 'SSE 连接中断/失败';
                rawAnswerText += `\n\n（错误：${msg}）`;
                renderAnswerWithImages(answerEl, rawAnswerText, []);
            } catch (_) {
                rawAnswerText += `\n\n（错误：SSE 连接中断/失败）`;
                renderAnswerWithImages(answerEl, rawAnswerText, []);
            }
            es.close();
            sendBtn.disabled = false;
        });
    } catch (e) {
        const bubble = botMsgEl.querySelector('.bubble');
        const progress = botMsgEl.querySelector('details.progress');
        const typing = botMsgEl.querySelector('.typing');
        if (typing) typing.remove();
        bubble.innerHTML = `请求失败：${escapeHtml(e.message || e)}\n\n` + (progress ? progress.outerHTML : '');
        sendBtn.disabled = false;
    }
}

sendBtn.addEventListener('click', onSend);

inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        onSend();
    }
});

btnClear.addEventListener('click', async () => {
    if (!confirm('确定要清空当前会话的历史记录吗？这将无法恢复。')) return;
    try {
        const res = await fetch(`${API_BASE}/history/${sessionId}`, { method: 'DELETE' });
        if (!res.ok) {
            console.error("Failed to clear history backend, status:", res.status);
            alert('服务端清空失败，仅清空本地显示');
        }
    } catch (e) {
        console.error("Failed to clear history backend", e);
        alert('服务端清空失败，仅清空本地显示');
    }

    const nodes = Array.from(chatEl.querySelectorAll('.msg'));
    for (let i = 1; i < nodes.length; i++) nodes[i].remove();
    scrollToBottom();
});


// ---------------- 知识库管理模块 (Import) ----------------
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#38bdf8';
    dropZone.style.background = 'rgba(56, 189, 248, 0.08)';
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '';
    dropZone.style.background = '';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '';
    dropZone.style.background = '';
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

function handleFiles(files) {
    Array.from(files).forEach(uploadFile);
}

function updateProgressUI(wrapper, percentage) {
    const bar = wrapper.querySelector('.progress-bar');
    const text = wrapper.querySelector('.progress-text');
    const safePercentage = Math.min(Math.max(percentage, 0), 100);

    bar.style.width = safePercentage + '%';
    text.textContent = Math.round(safePercentage) + '%';

    if (safePercentage >= 100) {
        bar.style.background = 'linear-gradient(90deg, #10b981, #34d399)';
    } else if (safePercentage < 10) {
        bar.style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)';
    } else {
        bar.style.background = 'linear-gradient(90deg, #3b82f6, #60a5fa)';
    }
}

function formatDuration(seconds) {
    if (seconds < 60) {
        return seconds.toFixed(1) + 's';
    }
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(0);
    return `${mins}m${secs}s`;
}

function normalizeLogText(text) {
    if (typeof text !== 'string') return String(text);
    return text;
}

function renderLogsAndCalculateProgress(itemEl, doneList, runningList, totalNodes, durations) {
    const logSummary = itemEl.querySelector('.log-details summary');
    const logListEl = itemEl.querySelector('.log-list');

    const done = Array.isArray(doneList) ? doneList : [];
    const running = Array.isArray(runningList) ? runningList : [];
    const durationMap = durations || {};

    let currentPercentage = (done.length / totalNodes) * 100;
    if (running.length > 0) {
        currentPercentage += (0.5 / totalNodes) * 100;
    }
    if (currentPercentage >= 100) {
        currentPercentage = 95;
    }

    let totalDuration = 0;
    for (const key in durationMap) {
        totalDuration += durationMap[key];
    }
    const totalDurationText = totalDuration > 0 ? ` | 总耗时: ${formatDuration(totalDuration)}` : '';

    logSummary.textContent = `📋 运行日志 (已完成: ${done.length}, 进行中: ${running.length}, 共 ${totalNodes} 步${totalDurationText})`;
    logListEl.innerHTML = '';

    for (const item of done) {
        if (item === 'upload_file') continue;

        const li = document.createElement('li');
        const nameSpan = document.createElement('span');
        nameSpan.textContent = `${normalizeLogText(item)} 节点执行完成`;
        li.appendChild(nameSpan);

        if (durationMap[item] !== undefined) {
            const durationSpan = document.createElement('span');
            durationSpan.className = 'log-duration';
            durationSpan.textContent = formatDuration(durationMap[item]);
            li.appendChild(durationSpan);
        }

        li.className = 'log-done';
        logListEl.appendChild(li);
    }

    for (const item of running) {
        const li = document.createElement('li');
        li.textContent = `正在处理 ${normalizeLogText(item)} 节点...`;
        li.className = 'log-running';
        logListEl.appendChild(li);
    }

    logListEl.scrollTop = logListEl.scrollHeight;
    return currentPercentage;
}

async function uploadFile(file) {
    const isPDF = file.name.toLowerCase().endsWith('.pdf');
    const totalNodes = isPDF ? 8 : 7;

    const id = 'file-' + Math.random().toString(36).substr(2, 8);
    const html = `
        <div class="file-item" id="${id}">
            <div class="file-info">
                <div class="file-header">
                    <span class="file-name">${file.name}</span>
                    <span class="file-size">${(file.size / 1024).toFixed(2)} KB</span>
                </div>
                <div class="progress-wrapper">
                    <div class="progress-bar-container">
                        <div class="progress-bar"></div>
                    </div>
                    <div class="progress-text">0%</div>
                </div>
                <details class="log-details" open>
                    <summary>准备上传...</summary>
                    <ul class="log-list"></ul>
                </details>
            </div>
            <div class="status-badge status-uploading">上传中...</div>
        </div>
    `;
    fileList.insertAdjacentHTML('afterbegin', html);

    const itemEl = document.getElementById(id);
    const statusBadge = itemEl.querySelector('.status-badge');
    const progressWrapper = itemEl.querySelector('.progress-wrapper');
    const logListEl = itemEl.querySelector('.log-list');

    const formData = new FormData();
    formData.append('file', file);

    try {
        progressWrapper.style.display = 'block';
        const uploadPercentage = (1 / totalNodes) * 100;
        updateProgressUI(progressWrapper, Math.min(uploadPercentage, 10));

        const initialLog = document.createElement('li');
        initialLog.textContent = "开始上传文件...";
        initialLog.className = "log-running";
        logListEl.appendChild(initialLog);

        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error('Upload failed');

        const result = await response.json();
        const taskId = result.task_id;

        updateProgressUI(progressWrapper, uploadPercentage);
        statusBadge.textContent = '处理中';
        statusBadge.className = 'status-badge status-processing';

        pollStatus(taskId, itemEl, totalNodes);
    } catch (error) {
        console.error(error);
        statusBadge.textContent = '上传失败';
        statusBadge.className = 'status-badge status-error';
        updateProgressUI(progressWrapper, 0);

        const errLog = document.createElement('li');
        errLog.textContent = "❌ 网络请求失败或服务器异常";
        errLog.style.color = "#ef4444";
        logListEl.appendChild(errLog);
    }
}

function pollStatus(taskId, itemEl, totalNodes) {
    const statusBadge = itemEl.querySelector('.status-badge');
    const progressWrapper = itemEl.querySelector('.progress-wrapper');
    const logDetails = itemEl.querySelector('.log-details');
    let stopped = false;

    const interval = setInterval(async () => {
        if (stopped) return;

        try {
            const res = await fetch(`${API_BASE}/status/${taskId}`);
            if (!res.ok) throw new Error('Status request failed');

            const data = await res.json();
            if (stopped) return;

            const calculatedPercentage = renderLogsAndCalculateProgress(itemEl, data.done_list, data.running_list, totalNodes, data.durations);

            if (data.status === 'completed') {
                stopped = true;
                clearInterval(interval);
                statusBadge.textContent = '已完成';
                statusBadge.className = 'status-badge status-completed';
                updateProgressUI(progressWrapper, 100);

                setTimeout(() => {
                    logDetails.open = false;
                }, 1000);

            } else if (data.status === 'failed') {
                stopped = true;
                clearInterval(interval);
                statusBadge.textContent = '失败';
                statusBadge.className = 'status-badge status-error';

                const logListEl = itemEl.querySelector('.log-list');
                const errLog = document.createElement('li');
                errLog.textContent = "❌ 工作流执行失败中止";
                errLog.style.color = "#ef4444";
                logListEl.appendChild(errLog);

            } else if (data.status === 'processing') {
                updateProgressUI(progressWrapper, calculatedPercentage);
            }
        } catch (e) {
            console.error('Polling error', e);
        }
    }, 1500);
}


// ---------------- 引用抽屉侧边栏交互逻辑 ----------------
const sourceDrawer = document.getElementById('sourceDrawer');
const drawerOverlay = document.getElementById('drawerOverlay');
const closeDrawerBtn = document.getElementById('closeDrawer');

window.openSourceDrawer = function(chunkId) {
    const chunk = currentSessionChunks[chunkId];
    if (!chunk) {
        console.warn(`[openSourceDrawer] 未在缓存中找到标号 [${chunkId}] 的切片信息。当前已缓存的切片 ID:`, Object.keys(currentSessionChunks));
        alert(`未在本地缓存中找到标号 [${chunkId}] 对应的原文切片（该消息可能产生于持久化支持添加前，重新提问即可查看最新切片详情）。`);
        return;
    }

    document.getElementById('drawerFileTitle').textContent = chunk.file_title || '未知来源';
    document.getElementById('drawerSectionTitle').textContent = chunk.section_title || '正文小节';
    document.getElementById('drawerChunkContent').innerHTML = formatChunkToHtml(chunk.content || '暂无内容');

    sourceDrawer.classList.add('open');
    drawerOverlay.classList.add('open');
};

function closeSourceDrawer() {
    sourceDrawer.classList.remove('open');
    drawerOverlay.classList.remove('open');
}

if (closeDrawerBtn) {
    closeDrawerBtn.addEventListener('click', closeSourceDrawer);
}
if (drawerOverlay) {
    drawerOverlay.addEventListener('click', closeSourceDrawer);
}


// ---------------- 初始化启动 ----------------
apiHealth();
loadHistory();
setTimeout(() => inputEl.focus(), 200);
// 设定定时心跳检查
setInterval(apiHealth, 10000);
