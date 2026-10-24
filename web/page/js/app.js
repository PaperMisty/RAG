// 掌柜智库一体化前端逻辑

// 1. 初始化变量与 API 基地址
const API_BASE = window.location.origin.startsWith('http') ? window.location.origin : 'http://127.0.0.1:8000';
let sessionId = localStorage.getItem('kb_session_id');
if (!sessionId) {
    sessionId = 'sess-' + Math.random().toString(36).slice(2) + Date.now().toString(36);
    localStorage.setItem('kb_session_id', sessionId);
}
let chunksByMessage = {}; // 按消息（task_id，历史则为 _id）隔离的引用切片缓存：各轮 chunk id 均从 1 起重编，全局单 map 必然串轮
let currentTaskId = null;

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
        } else if (targetTab === 'tab-observer') {
            if (typeof loadObserverFolders === 'function') {
                loadObserverFolders(true);
            }
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
function linkCitations(mdText, messageKey) {
    const seqByChunkId = new Map();
    const safeKey = String(messageKey || '').replace(/'/g, "\\'");
    return mdText.replace(/\[\[(\d+)\]\]/g, (match, chunkId) => {
        if (!seqByChunkId.has(chunkId)) {
            seqByChunkId.set(chunkId, seqByChunkId.size + 1);
        }
        return `<span class="citation-badge" onclick="openSourceDrawer('${safeKey}', '${chunkId}')">[${seqByChunkId.get(chunkId)}]</span>`;
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

function formatAnswerToHtml(answerText, messageKey) {
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
    mdText = linkCitations(mdText, messageKey);

    // 调用 marked 解析库解析 Markdown，如果未引入则 fallback 降级为普通文本
    let html = '';
    if (typeof marked !== 'undefined' && typeof marked.parse === 'function') {
        html = marked.parse(mdText);
    } else {
        html = escapeHtml(mdText).replace(/\r?\n/g, '<br>');
    }

    return html;
}

function renderAnswerWithImages(containerEl, answerText, candidateImageUrls, messageKey) {
    let htmlContent = formatAnswerToHtml(answerText || '', messageKey);

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

function addBotMsgWithTime(text, ts, imageUrls, messageKey) {
    const id = 'bot-his-' + Math.random().toString(36).slice(2);
    const traceBtnHtml = (messageKey && !String(messageKey).startsWith('his-'))
        ? `<button class="btn-jump-trace" onclick="jumpToTrace('${escapeHtml(messageKey)}')">🔍 查看 Trace</button>`
        : '';
    const html = `
    <div class="msg bot" id="${id}">
      <div class="avatar bot">🤖</div>
      <div>
        <div class="bubble"><div class="answer"></div></div>
        <div class="meta">${formatTime(ts)}${traceBtnHtml}</div>
      </div>
    </div>
  `;
    chatEl.insertAdjacentHTML('beforeend', html);
    const el = document.getElementById(id);
    if (el) {
        const answerEl = el.querySelector('.answer');
        renderAnswerWithImages(answerEl, text || '', imageUrls || [], messageKey);
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

function renderProgress(botMsgEl, doneList, runningList, status, elapsed_s, stalled) {
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

    // Elapsed / stall display: nodes used to sit silent for 45-70 s
    // with only the typing dots changing, which reads as a hang.
    // The summary now always carries elapsed time, and once no event
    // has arrived for 10 s it says so explicitly instead of staying quiet.
    let summaryText = `阶段进度（已完成 ${totalDone}，进行中 ${totalRun}，状态：${displayStatus}`;
    if (typeof elapsed_s === 'number') {
        summaryText += `，已等待 ${Math.floor(elapsed_s)} 秒`;
    }
    if (stalled) {
        summaryText += '，仍在处理';
    }
    summaryText += '）';
    summary.textContent = summaryText;

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
        
        chunksByMessage = {};
        for (let hi = 0; hi < items.length; hi++) {
            const item = items[hi];
            if (item.role === 'user') {
                addUserMsgWithTime(item.text || '', item.ts);
            } else {
                const messageKey = item.task_id || item._id || ('his-' + hi);
                if (Array.isArray(item.chunks)) {
                    const m = {};
                    item.chunks.forEach(c => { m[c.id] = c; });
                    chunksByMessage[messageKey] = m;
                }
                addBotMsgWithTime(item.text || '', item.ts, item.image_urls || [], messageKey);
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
    // 新提问只建本轮的缓存键，不清空旧轮：旧轮角标仍需按本轮 task_id 之外的键解析
    inputEl.value = '';
    addUserMsg(text);
    const botMsgEl = addBotMsgSkeleton();
    sendBtn.disabled = true;

    try {
        const data = await submitQuery(text);
        currentTaskId = data.task_id || null;
        if (currentTaskId) chunksByMessage[currentTaskId] = {};
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
        // Waiting timer: the backend now heartbeats every 5 s, but the
        // timer also covers a stalled connection (no event at all).
        // It only rewrites the progress summary; it never aborts a
        // query, because normal queries legitimately run past 2 minutes.
        const taskStartAt = Date.now();
        let lastEventAt = taskStartAt;
        let lastDone = [];
        let lastRunning = [];
        let lastStatus = 'pending';
        const markEvent = () => { lastEventAt = Date.now(); };
        const elapsedSec = () => (Date.now() - taskStartAt) / 1000;
        const waitTimer = setInterval(() => {
            const stalled = (Date.now() - lastEventAt) / 1000 >= 10;
            renderProgress(botMsgEl, lastDone, lastRunning, lastStatus, elapsedSec(), stalled);
        }, 5000);
        const stopWaiting = () => { if (waitTimer) clearInterval(waitTimer); };

        es.addEventListener('chunks', (e) => {
            markEvent();
            try {
                const chunks = JSON.parse(e.data || '[]');
                if (currentTaskId) {
                    if (!chunksByMessage[currentTaskId]) chunksByMessage[currentTaskId] = {};
                    chunks.forEach(c => {
                        chunksByMessage[currentTaskId][c.id] = c;
                    });
                }
            } catch (_) {}
        });

        es.addEventListener('heartbeat', (e) => {
            markEvent();
            try {
                const d = JSON.parse(e.data || '{}');
                lastDone = d.done_list || lastDone;
                lastRunning = d.running_list || lastRunning;
                lastStatus = d.status || lastStatus;
                const elapsed = (typeof d.elapsed_s === 'number') ? d.elapsed_s : elapsedSec();
                renderProgress(botMsgEl, lastDone, lastRunning, lastStatus, elapsed, true);
            } catch (_) {}
        });

        es.addEventListener('progress', (e) => {
            markEvent();
            try {
                const d = JSON.parse(e.data || '{}');
                lastDone = d.done_list || [];
                lastRunning = d.running_list || [];
                lastStatus = d.status || lastStatus;
                renderProgress(botMsgEl, lastDone, lastRunning, lastStatus, elapsedSec(), false);
                if (d && d.status === 'completed') {
                    stopWaiting();
                    const typing = botMsgEl.querySelector('.typing');
                    if (typing) typing.remove();
                    sendBtn.disabled = false;
                }
            } catch (_) {}
        });

        es.addEventListener('delta', (e) => {
            markEvent();
            try {
                const d = JSON.parse(e.data || '{}');
                const delta = d.delta || '';
                if (delta) {
                    rawAnswerText += delta;
                    renderAnswerWithImages(answerEl, rawAnswerText, [], currentTaskId);
                    scrollToBottom();
                }
            } catch (_) {}
        });

        es.addEventListener('final', (e) => {
            stopWaiting();
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
                renderAnswerWithImages(answerEl, finalText, d.image_urls || [], currentTaskId);
            } catch (_) {}

            if (currentTaskId) {
                const metaEl = botMsgEl.querySelector('.meta');
                if (metaEl && !metaEl.querySelector('.btn-jump-trace')) {
                    metaEl.insertAdjacentHTML(
                        'beforeend',
                        `<button class="btn-jump-trace" onclick="jumpToTrace('${escapeHtml(currentTaskId)}')">🔍 查看 Trace</button>`
                    );
                }
            }
            
            es.close();
            sendBtn.disabled = false;
            scrollToBottom();
        });

        es.addEventListener('error', (e) => {
            stopWaiting();
            const typing = botMsgEl.querySelector('.typing');
            if (typing) typing.remove();

            const progress = botMsgEl.querySelector('details.progress');
            if (progress) progress.removeAttribute('open');

            // The backend now puts one readable sentence in data.error
            // (see main_service._readable_error); a bare browser
            // EventSource error has no data and keeps the generic text.
            // Never render a raw traceback or credential here.
            let msg = 'SSE 连接中断/失败';
            try {
                if (e && e.data) {
                    const parsed = JSON.parse(e.data);
                    if (parsed && parsed.error) msg = parsed.error;
                }
            } catch (_) {}
            if (rawAnswerText) {
                rawAnswerText += `\n\n（错误：${msg}）`;
            } else {
                rawAnswerText = `（错误：${msg}）`;
            }
            renderAnswerWithImages(answerEl, rawAnswerText, [], currentTaskId);
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

window.openSourceDrawer = function(messageKey, chunkId) {
    // 兼容旧式单参调用 openSourceDrawer(chunkId)：先按当前轮找，再兜底全量找；正常路径（带消息键）不走兜底。
    if (chunkId === undefined) { chunkId = messageKey; messageKey = null; }
    let chunk = null;
    if (messageKey && chunksByMessage[messageKey]) chunk = chunksByMessage[messageKey][chunkId];
    if (!chunk && currentTaskId && chunksByMessage[currentTaskId]) chunk = chunksByMessage[currentTaskId][chunkId];
    if (!chunk) {
        for (const k of Object.keys(chunksByMessage)) {
            if (chunksByMessage[k] && chunksByMessage[k][chunkId]) { chunk = chunksByMessage[k][chunkId]; break; }
        }
    }
    if (!chunk) {
        console.warn(`[openSourceDrawer] 未在缓存中找到消息 [${messageKey}] 的标号 [${chunkId}] 切片信息。`);
        alert(`未在本地缓存中找到该轮标号 [${chunkId}] 对应的原文切片（该消息可能产生于持久化支持添加前，重新提问即可查看最新切片详情）。`);
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


// ====================================================================
// ---------------- RAG 链路观测层 (Trace Observer) 逻辑 ----------------
// ====================================================================

const obsFolderListEl = document.getElementById('obsFolderList');
const obsCanvasEl = document.getElementById('obsCanvas');
const obsSummaryPillEl = document.getElementById('obsSummaryPill');
const obsSearchInputEl = document.getElementById('obsSearchInput');
const btnRefreshTracesEl = document.getElementById('btnRefreshTraces');

let cachedObserverFolders = [];
let selectedTraceId = null;
let selectedFolderHour = null;

function formatMs(ms) {
    const val = Number(ms || 0);
    if (val <= 0) return '0.0ms';
    if (val < 1000) return `${val.toFixed(1)}ms`;
    return `${(val / 1000).toFixed(2)}s (${Math.round(val)}ms)`;
}

function getRouteLabelText(routeKey) {
    const map = {
        'high_hybrid': '高置信·混合检索',
        'medium_clarify': '中置信·澄清反问',
        'low_web_only': '低置信·网络搜索',
    };
    return map[routeKey] || routeKey || '未知路由';
}

// 从对话气泡一键跳转到第三栏观测层并打开指定 Trace
window.jumpToTrace = async function(traceId) {
    if (!traceId) return;
    navItems.forEach(nav => nav.classList.remove('active'));
    tabPanes.forEach(pane => pane.classList.remove('active'));

    const obsNav = document.querySelector('.nav-item[data-tab="tab-observer"]');
    const obsPane = document.getElementById('tab-observer');
    if (obsNav) obsNav.classList.add('active');
    if (obsPane) obsPane.classList.add('active');

    selectedTraceId = traceId;
    await loadObserverFolders(true);
    await selectTrace(traceId, null);
};

async function loadObserverFolders(preserveSelection = true) {
    if (!obsFolderListEl) return;
    try {
        const res = await fetch(`${API_BASE}/observer/folders`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        cachedObserverFolders = Array.isArray(data.folders) ? data.folders : [];

        if (obsSummaryPillEl) {
            obsSummaryPillEl.textContent = `共 ${data.total_folders || 0} 个时段 · ${data.total_traces || 0} 条 Trace`;
        }

        renderObserverFolderTree(obsSearchInputEl ? obsSearchInputEl.value : '');

        // 若尚未选中任何 Trace 且列表非空，默认自动加载最新一条 Trace
        if ((!preserveSelection || !selectedTraceId) && cachedObserverFolders.length > 0) {
            const firstFolder = cachedObserverFolders[0];
            if (firstFolder.traces && firstFolder.traces.length > 0) {
                const latest = firstFolder.traces[0];
                await selectTrace(latest.trace_id, firstFolder.folder_hour);
            }
        }
    } catch (e) {
        console.error('加载观测层目录失败:', e);
        obsFolderListEl.innerHTML = `<div class="obs-empty-hint">加载失败: ${escapeHtml(e.message || e)}</div>`;
    }
}

function renderObserverFolderTree(keyword = '') {
    if (!obsFolderListEl) return;
    const kw = String(keyword || '').trim().toLowerCase();

    if (!cachedObserverFolders || cachedObserverFolders.length === 0) {
        obsFolderListEl.innerHTML = `<div class="obs-empty-hint">暂无 Trace 归档记录<br><small>在“智能对话”发送问题后将自动按 yy-mm-dd:hh 归档</small></div>`;
        return;
    }

    let html = '';
    cachedObserverFolders.forEach((folderObj, fIdx) => {
        const folderName = folderObj.folder_hour || 'unknown';
        const traces = Array.isArray(folderObj.traces) ? folderObj.traces : [];
        const filtered = traces.filter(t => {
            if (!kw) return true;
            const hay = `${t.original_query} ${t.rewritten_query} ${(t.item_names || []).join(' ')} ${t.trace_id}`.toLowerCase();
            return hay.includes(kw);
        });

        if (filtered.length === 0) return;

        const hasSelected = filtered.some(t => t.trace_id === selectedTraceId);
        const isOpen = fIdx === 0 || hasSelected || kw.length > 0;

        html += `
        <div class="obs-folder-group ${isOpen ? 'open' : ''}">
            <div class="obs-folder-header" onclick="this.parentElement.classList.toggle('open')">
                <div class="obs-folder-title">
                    <span>📁</span>
                    <span>${escapeHtml(folderName)}</span>
                </div>
                <span class="obs-folder-count">${filtered.length}</span>
            </div>
            <div class="obs-folder-items">
        `;

        filtered.forEach(item => {
            const isActive = item.trace_id === selectedTraceId;
            const timePart = (item.created_at || '').split(' ')[1] || item.created_at || '';
            const scoreStr = typeof item.confidence_score === 'number' ? item.confidence_score.toFixed(2) : '0.00';
            const routeCls = item.route_taken || 'high_hybrid';
            const safeId = escapeHtml(item.trace_id);
            const safeFolder = escapeHtml(folderName);

            html += `
                <div class="obs-trace-item ${isActive ? 'active' : ''}" data-trace-id="${safeId}" onclick="selectTrace('${safeId}', '${safeFolder}')">
                    <div class="obs-trace-query" title="${escapeHtml(item.original_query || '')}">
                        ${escapeHtml(item.original_query || '（空查询）')}
                    </div>
                    <div class="obs-trace-meta">
                        <span>🕒 ${escapeHtml(timePart)}</span>
                        <span class="obs-route-tag ${escapeHtml(routeCls)}">${escapeHtml(getRouteLabelText(routeCls))} (${scoreStr})</span>
                        <span>⏱ ${formatMs(item.total_duration_ms)}</span>
                    </div>
                </div>
            `;
        });

        html += `</div></div>`;
    });

    obsFolderListEl.innerHTML = html || `<div class="obs-empty-hint">未找到匹配 "${escapeHtml(keyword)}" 的 Trace</div>`;
}

window.selectTrace = async function(traceId, folderHour) {
    if (!traceId || !obsCanvasEl) return;
    selectedTraceId = traceId;
    if (folderHour) selectedFolderHour = folderHour;

    // 更新左侧高亮态
    document.querySelectorAll('.obs-trace-item').forEach(el => {
        if (el.getAttribute('data-trace-id') === traceId) {
            el.classList.add('active');
        } else {
            el.classList.remove('active');
        }
    });

    obsCanvasEl.innerHTML = `<div class="obs-placeholder"><div class="obs-placeholder-icon">⏳</div><h3>正在加载 Trace (${escapeHtml(traceId.slice(0, 8))}...)</h3></div>`;

    try {
        const qParam = folderHour ? `?folder=${encodeURIComponent(folderHour)}` : '';
        const res = await fetch(`${API_BASE}/observer/trace/${encodeURIComponent(traceId)}${qParam}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const traceData = await res.json();
        renderTraceCanvas(traceData);
    } catch (e) {
        console.error('加载单条 Trace 详情失败:', e);
        obsCanvasEl.innerHTML = `<div class="obs-placeholder"><div class="obs-placeholder-icon">⚠️</div><h3>无法加载 Trace 详情</h3><p>${escapeHtml(e.message || e)}</p></div>`;
    }
};

// 渲染每个模块方格正下方的耗时与网络 IO 铭牌
function renderModuleTimingBar(timingObj, fallbackDesc = '') {
    const t = timingObj || {};
    const totalMs = Number(t.total_ms || 0);
    const ioList = Array.isArray(t.io_breakdown) ? t.io_breakdown : [];

    let chipsHtml = '';
    if (ioList.length > 0) {
        ioList.forEach(io => {
            const ioType = io.io_type || 'network';
            const iconMap = {
                'network': '🌐',
                'llm': '🤖',
                'db': '🗄️',
                'cpu': '⚡',
            };
            const icon = iconMap[ioType] || '🌐';
            const detailStr = io.detail ? ` (${io.detail})` : '';
            chipsHtml += `
                <span class="timing-io-chip io-${escapeHtml(ioType)}" title="${escapeHtml(io.name + detailStr)}">
                    ${icon} ${escapeHtml(io.name)}: <strong>${formatMs(io.duration_ms)}</strong>${escapeHtml(detailStr)}
                </span>
            `;
        });
    } else if (fallbackDesc) {
        chipsHtml = `<span class="timing-io-chip io-cpu">ℹ️ ${escapeHtml(fallbackDesc)}</span>`;
    } else {
        chipsHtml = `<span class="timing-io-chip io-cpu">⚡ 无远程网络 IO</span>`;
    }

    return `
        <div class="trace-module-timing-bar">
            <div class="timing-total-badge">
                ⏱ 模块总耗时: ${formatMs(totalMs)}
            </div>
            <div class="timing-io-list">
                ${chipsHtml}
            </div>
        </div>
    `;
}

// 渲染模块之间的虚线连接器
function renderDashedConnector(label = '', isGrayed = false) {
    const labelHtml = label ? `<span class="connector-label">${escapeHtml(label)}</span>` : '';
    return `
        <div class="trace-dashed-connector ${isGrayed ? 'is-grayed' : ''}">
            <div class="dashed-line"></div>
            ${labelHtml}
            <div class="connector-arrow">▼</div>
        </div>
    `;
}

// 批量展开/收拢某个模块下的所有 Chunk 卡片
window.toggleAllChunksInBox = function(btnEl, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    const cards = container.querySelectorAll('.chunk-card');
    const anyCollapsed = Array.from(cards).some(c => !c.classList.contains('is-expanded'));
    cards.forEach(c => {
        if (anyCollapsed) {
            c.classList.add('is-expanded');
        } else {
            c.classList.remove('is-expanded');
        }
    });
    btnEl.textContent = anyCollapsed ? '🔼 全部收拢' : '🔽 全部展开';
};

// 渲染可展开/收拢的排名 Chunks 列表（支持 RRF 灰化、断崖分割线与断崖灰化）
function renderChunkListHtml(chunks, listId, mode = 'standard', extraConfig = {}) {
    if (!Array.isArray(chunks) || chunks.length === 0) {
        return `<div class="obs-empty-hint" style="padding:18px;">该阶段无切片数据（未触发或返回为空）</div>`;
    }

    let html = `<div class="chunk-rank-list" id="${escapeHtml(listId)}">`;
    let cliffDividerInserted = false;

    chunks.forEach((ch, idx) => {
        const rank = ch.rank || (idx + 1);
        const isGray = Boolean(ch.is_grayed);
        const scoreVal = typeof ch.score === 'number' ? ch.score.toFixed(4) : '0.0000';
        const title = ch.file_title || ch.item_name || '未知来源';
        const section = ch.section_title ? ` · ${ch.section_title}` : '';
        const rawContent = String(ch.content || '');
        const preview = rawContent.replace(/\s+/g, ' ').slice(0, 75) + (rawContent.length > 75 ? '...' : '');

        // 断崖检测模式下：渲染相邻切片分差气泡 & 断崖红色虚线分割线
        if (mode === 'cliff' && idx > 0) {
            const absGap = typeof ch.abs_gap === 'number' ? ch.abs_gap.toFixed(4) : '-';
            const ratioGap = typeof ch.ratio_gap === 'number' ? (ch.ratio_gap * 100).toFixed(1) + '%' : '-';
            html += `
                <div class="cliff-gap-pill-row">
                    <span class="cliff-gap-pill">↕ 相邻分差: Δabs = ${absGap} | Δratio = ${ratioGap}</span>
                </div>
            `;
        }

        if (mode === 'cliff' && isGray && !cliffDividerInserted) {
            cliffDividerInserted = true;
            const reasonText = extraConfig.trigger_reason || ch.gray_reason || '达到断崖阈值，后续切片被截断淘汰';
            html += `
                <div class="cliff-cutoff-divider">
                    <span>✂️ 断崖截断线：${escapeHtml(reasonText)}</span>
                    <span>以下切片标记为灰色（不送入大模型）</span>
                </div>
            `;
        }

        if (mode === 'rrf' && isGray && !cliffDividerInserted) {
            cliffDividerInserted = true;
            const topK = extraConfig.top_k || 10;
            html += `
                <div class="cliff-cutoff-divider" style="border-color:#94a3b8;background:#f1f5f9;color:#475569;">
                    <span>📉 RRF Top-${escapeHtml(String(topK))} 截断线</span>
                    <span>以下切片被挤在 Top-${escapeHtml(String(topK))} 后面（标记灰色）</span>
                </div>
            `;
        }

        // 来源排名辅助角标（如在 RRF 或 Reranker 中展示前序排名）
        let srcBadges = '';
        if (mode === 'rrf') {
            if (ch.dense_rank) srcBadges += `<span class="chunk-src-rank-tag">Dense #${ch.dense_rank}</span>`;
            if (ch.sparse_rank) srcBadges += `<span class="chunk-src-rank-tag">Sparse #${ch.sparse_rank}</span>`;
            if (ch.hyde_rank) srcBadges += `<span class="chunk-src-rank-tag">HyDE #${ch.hyde_rank}</span>`;
        } else if ((mode === 'reranker' || mode === 'cliff') && ch.rrf_rank) {
            const diff = ch.rrf_rank - rank;
            const diffStr = diff > 0 ? `↑+${diff}` : (diff < 0 ? `↓${diff}` : '=');
            srcBadges += `<span class="chunk-src-rank-tag">RRF #${ch.rrf_rank} (${diffStr})</span>`;
        }

        let grayBadge = '';
        if (isGray) {
            const badgeText = mode === 'rrf' ? '未进 Top-K (已置灰)' : '已断崖淘汰 (已置灰)';
            grayBadge = `<span class="chunk-gray-tag">${escapeHtml(badgeText)}</span>`;
        }

        html += `
            <div class="chunk-card ${isGray ? 'is-grayed' : ''}">
                <div class="chunk-card-header" onclick="this.parentElement.classList.toggle('is-expanded')">
                    <span class="chunk-rank-badge">#${rank}</span>
                    <div class="chunk-main-info">
                        <div class="chunk-title-row">
                            <span class="chunk-title-text">${escapeHtml(title)}${escapeHtml(section)}</span>
                            ${srcBadges}
                            ${grayBadge}
                        </div>
                        <div class="chunk-preview-text">${escapeHtml(preview || '（无正文内容）')}</div>
                    </div>
                    <div class="chunk-right-meta">
                        <span class="chunk-score-badge">score: ${scoreVal}</span>
                        <span class="chunk-expand-icon">▼</span>
                    </div>
                </div>
                <div class="chunk-card-detail">
                    <div class="chunk-detail-meta-bar">
                        <span>🆔 Chunk ID: ${escapeHtml(ch.chunk_id || '-')}</span>
                        <span>🏷️ 主体: ${escapeHtml(ch.item_name || '-')}</span>
                        <span>📂 来源: ${escapeHtml(ch.source || 'local')}</span>
                        ${ch.gray_reason ? `<span>⚠️ 灰化原因: ${escapeHtml(ch.gray_reason)}</span>` : ''}
                    </div>
                    <div class="md-content">${formatChunkToHtml(rawContent)}</div>
                </div>
            </div>
        `;
    });

    html += `</div>`;
    return html;
}

function renderTraceCanvas(trace) {
    if (!obsCanvasEl || !trace) return;

    const qMod = trace.query_module || {};
    const rMod = trace.router_module || {};
    const retMod = trace.retrieval_module || {};
    const rrfMod = trace.rrf_module || {};
    const rerankMod = trace.reranker_module || {};
    const cliffMod = trace.cliff_module || {};
    const outMod = trace.output_module || {};

    const routeTaken = rMod.route_taken || 'high_hybrid';
    const isHighRoute = routeTaken === 'high_hybrid';
    const confScore = Number(rMod.confidence_score || 0);
    const confPct = Math.min(Math.max(confScore * 100, 0), 100);
    const barColor = confScore >= 0.8 ? '#16a34a' : (confScore > 0.6 ? '#f59e0b' : '#64748b');

    // 1. 关键词与主体胶囊
    const kwPills = (qMod.key_words || []).length > 0
        ? qMod.key_words.map(k => `<span class="obs-tag-pill kw-pill">🔑 ${escapeHtml(k)}</span>`).join('')
        : `<span class="obs-tag-pill raw-item-pill">无提取关键词</span>`;

    const alignedItems = (qMod.aligned_item_names || []).length > 0
        ? qMod.aligned_item_names.map(n => `<span class="obs-tag-pill item-pill">✅ ${escapeHtml(n)}</span>`).join('')
        : `<span class="obs-tag-pill raw-item-pill">未对齐到高置信知识库主体</span>`;

    const rawItems = (qMod.extracted_item_names || []).length > 0
        ? qMod.extracted_item_names.map(n => `<span class="obs-tag-pill raw-item-pill">LLM提取: ${escapeHtml(n)}</span>`).join('')
        : '';

    // 2. 候选主体得分表
    const candidates = Array.isArray(rMod.matched_candidates) ? rMod.matched_candidates : [];
    const candPills = candidates.length > 0
        ? candidates.slice(0, 6).map(c => `<span class="obs-tag-pill ${c.score >= 0.8 ? 'item-pill' : 'raw-item-pill'}">${escapeHtml(c.item_name)}: ${Number(c.score).toFixed(3)}</span>`).join('')
        : `<span class="obs-tag-pill raw-item-pill">Milvus 无候选匹配项</span>`;

    // 3. 三条路由线渲染（命中路线高亮，其余两条置灰）
    const routesList = Array.isArray(rMod.routes) && rMod.routes.length > 0 ? rMod.routes : [
        { route_key: 'high_hybrid', label: '高置信度路由 (score ≥ 0.8)', condition: '走混合检索 (Dense + Sparse + HyDE)', is_active: routeTaken === 'high_hybrid', target_desc: '进入 Dense/Sparse 检索 → RRF → Reranker → 断崖检测' },
        { route_key: 'medium_clarify', label: '中置信度路由 (0.6 < score < 0.8)', condition: '多候选主体澄清反问', is_active: routeTaken === 'medium_clarify', target_desc: '跳过向量切片检索，直接生成澄清选项卡' },
        { route_key: 'low_web_only', label: '低置信度路由 (score ≤ 0.6)', condition: '纯网络搜索兜底', is_active: routeTaken === 'low_web_only', target_desc: '跳过本地知识库，仅走 Tavily 网络搜索回答' },
    ];

    const branchesHtml = routesList.map(b => {
        const activeCls = b.is_active ? 'is-active' : 'is-grayed';
        const statusText = b.is_active ? '⚡ 实际执行路由' : '⚪ 未命中 (已置灰)';
        return `
            <div class="router-branch-col ${activeCls}">
                <div class="router-branch-line-top"></div>
                <div class="router-branch-card">
                    <div class="router-branch-head">
                        <span>${escapeHtml(b.label)}</span>
                        <span class="router-status-pill">${statusText}</span>
                    </div>
                    <div class="router-branch-desc">
                        <div><strong>条件：</strong>${escapeHtml(b.condition || '')}</div>
                        <div style="margin-top:4px;"><strong>流向：</strong>${escapeHtml(b.target_desc || '')}</div>
                    </div>
                </div>
                <div class="router-branch-line-bottom"></div>
            </div>
        `;
    }).join('');

    const html = `
    <div class="obs-flow-container">
        <!-- 顶部概览横幅 -->
        <div class="trace-banner">
            <div class="trace-banner-left">
                <span class="trace-folder-badge">📁 归档目录: ${escapeHtml(trace.folder_hour || '-')}</span>
                <span class="trace-id-badge">🆔 Trace: ${escapeHtml(trace.trace_id || '-')}</span>
                <span class="trace-id-badge">🕒 ${escapeHtml(trace.created_at || '-')}</span>
            </div>
            <div class="trace-total-time-badge">
                🏁 端到端总耗时: ${formatMs(trace.total_duration_ms)}
            </div>
        </div>

        <!-- ==================== 模块 1：Query 理解与改写 ==================== -->
        <div class="trace-module-wrap">
            <div class="trace-module-box">
                <div class="trace-module-header">
                    <div class="trace-module-title">
                        <span class="trace-step-num">1</span>
                        <span>Query 理解、改写与主体提取模块 (Query & Intent Extraction)</span>
                    </div>
                    <span class="obs-tag-pill raw-item-pill">历史上下文: ${escapeHtml(String(qMod.history_turns || 0))} 条</span>
                </div>
                <div class="trace-module-body">
                    <div class="query-fields-grid">
                        <div class="query-field-card">
                            <div class="query-field-label"><span>💬 原始 Query (original_query)</span></div>
                            <div class="query-field-val">${escapeHtml(qMod.original_query || '-')}</div>
                        </div>
                        <div class="query-field-card">
                            <div class="query-field-label"><span>✍️ 改写后 Query (rewritten_query)</span></div>
                            <div class="query-field-val" style="color:#92400e;">${escapeHtml(qMod.rewritten_query || '-')}</div>
                        </div>
                        <div class="query-field-card">
                            <div class="query-field-label"><span>🔑 提取关键词 (key_words)</span></div>
                            <div class="tag-pills-wrap">${kwPills}</div>
                        </div>
                        <div class="query-field-card">
                            <div class="query-field-label"><span>🏷️ 识别商品主体 (item_name)</span></div>
                            <div class="tag-pills-wrap">${alignedItems} ${rawItems}</div>
                        </div>
                    </div>
                </div>
            </div>
            ${renderModuleTimingBar(qMod.timing)}
        </div>

        ${renderDashedConnector('虚线传递: rewritten_query + item_names')}

        <!-- ==================== 模块 2：置信度评估与三路路由线 ==================== -->
        <div class="trace-module-wrap">
            <div class="trace-module-box">
                <div class="trace-module-header">
                    <div class="trace-module-title">
                        <span class="trace-step-num">2</span>
                        <span>主体置信度分数与实际路由分支 (Confidence Score & Routing Lines)</span>
                    </div>
                    <span class="obs-route-tag ${escapeHtml(routeTaken)}" style="font-size:12px;padding:3px 10px;">
                        当前激活: ${escapeHtml(getRouteLabelText(routeTaken))}
                    </span>
                </div>
                <div class="trace-module-body">
                    <div class="confidence-meter-row">
                        <div class="confidence-score-big">
                            <span class="score-label">最高匹配置信度分数</span>
                            <span class="score-num" style="color:${barColor};">${confScore.toFixed(4)}</span>
                        </div>
                        <div class="confidence-bar-wrap">
                            <div class="confidence-bar-track">
                                <div class="confidence-bar-fill" style="width:${confPct}%;background:${barColor};"></div>
                            </div>
                            <div class="confidence-scale-marks">
                                <span>0.0 (无匹配)</span>
                                <span>0.6 (低/中置信度阈值)</span>
                                <span>0.8 (高置信度阈值)</span>
                                <span>1.0</span>
                            </div>
                        </div>
                    </div>
                    <div style="margin-bottom:12px;">
                        <span style="font-size:12px;font-weight:700;color:#8d7a6b;margin-right:8px;">📊 Milvus 候选物料得分：</span>
                        <div class="tag-pills-wrap" style="display:inline-flex;">${candPills}</div>
                    </div>
                    <!-- 三条虚线路由分支（命中分支高亮，其余两条灰色） -->
                    <div class="router-branches-grid">
                        ${branchesHtml}
                    </div>
                </div>
            </div>
            ${renderModuleTimingBar(rMod.timing)}
        </div>

        ${renderDashedConnector(isHighRoute ? '高置信混合检索路由激活 (双路并行召回)' : '当前路由跳过本地向量检索 (下游检索模块置灰)', !isHighRoute)}

        <!-- ==================== 模块 3：Dense 检索与 Sparse 检索并列模块 ==================== -->
        <div class="dual-retrieval-grid">
            <!-- 3A: Dense 检索出来的 Chunks -->
            <div class="trace-module-wrap ${isHighRoute ? '' : 'is-inactive-module'}">
                <div class="trace-module-box">
                    <div class="trace-module-header">
                        <div class="trace-module-title">
                            <span class="trace-step-num">3A</span>
                            <span>Dense 稠密检索 Chunks (按 Cosine 排名)</span>
                        </div>
                        <div class="trace-module-actions">
                            <span class="obs-folder-count">${(retMod.dense_chunks || []).length} 条</span>
                            <button class="btn-fold-toggle" onclick="toggleAllChunksInBox(this, 'obs-dense-list')">🔽 全部展开</button>
                        </div>
                    </div>
                    <div class="trace-module-body">
                        ${renderChunkListHtml(retMod.dense_chunks || [], 'obs-dense-list', 'dense')}
                    </div>
                </div>
                ${renderModuleTimingBar(retMod.dense_timing, '未执行 Dense 检索')}
            </div>

            <!-- 3B: Sparse 检索出来的 Chunks -->
            <div class="trace-module-wrap ${isHighRoute ? '' : 'is-inactive-module'}">
                <div class="trace-module-box">
                    <div class="trace-module-header">
                        <div class="trace-module-title">
                            <span class="trace-step-num">3B</span>
                            <span>Sparse 稀疏检索 Chunks (按 IP 排名)</span>
                        </div>
                        <div class="trace-module-actions">
                            <span class="obs-folder-count">${(retMod.sparse_chunks || []).length} 条</span>
                            <button class="btn-fold-toggle" onclick="toggleAllChunksInBox(this, 'obs-sparse-list')">🔽 全部展开</button>
                        </div>
                    </div>
                    <div class="trace-module-body">
                        ${renderChunkListHtml(retMod.sparse_chunks || [], 'obs-sparse-list', 'sparse')}
                    </div>
                </div>
                ${renderModuleTimingBar(retMod.sparse_timing, '未执行 Sparse 检索')}
            </div>
        </div>

        ${renderDashedConnector('双路召回切片汇入 RRF 倒排融合 (k=60)', !rrfMod.executed)}

        <!-- ==================== 模块 4：RRF 之后的 Top-K 模块 ==================== -->
        <div class="trace-module-wrap ${rrfMod.executed ? '' : 'is-inactive-module'}">
            <div class="trace-module-box">
                <div class="trace-module-header">
                    <div class="trace-module-title">
                        <span class="trace-step-num">4</span>
                        <span>RRF 倒排融合 Top-${escapeHtml(String(rrfMod.top_k || 10))} (被挤在 Top-K 后面的标记灰色)</span>
                    </div>
                    <div class="trace-module-actions">
                        <span class="obs-tag-pill item-pill">入围 Top-K: ${rrfMod.kept_count || 0} 条</span>
                        <span class="obs-tag-pill raw-item-pill">尾部置灰: ${rrfMod.grayed_count || 0} 条</span>
                        <button class="btn-fold-toggle" onclick="toggleAllChunksInBox(this, 'obs-rrf-list')">🔽 全部展开</button>
                    </div>
                </div>
                <div class="trace-module-body">
                    ${renderChunkListHtml(rrfMod.chunks || [], 'obs-rrf-list', 'rrf', { top_k: rrfMod.top_k || 10 })}
                </div>
            </div>
            ${renderModuleTimingBar(rrfMod.timing, '未执行 RRF 模块')}
        </div>

        ${renderDashedConnector('RRF Top-K 切片 + Web 补充切片送入 Cross-Encoder 重排', !rerankMod.executed)}

        <!-- ==================== 模块 5：Reranker 重排的 Chunks 模块 ==================== -->
        <div class="trace-module-wrap ${rerankMod.executed ? '' : 'is-inactive-module'}">
            <div class="trace-module-box">
                <div class="trace-module-header">
                    <div class="trace-module-title">
                        <span class="trace-step-num">5</span>
                        <span>Reranker 交叉编码重排 Chunks (按重排得分排名)</span>
                    </div>
                    <div class="trace-module-actions">
                        <span class="obs-folder-count">共 ${(rerankMod.chunks || []).length} 条参与重排</span>
                        <button class="btn-fold-toggle" onclick="toggleAllChunksInBox(this, 'obs-rerank-list')">🔽 全部展开</button>
                    </div>
                </div>
                <div class="trace-module-body">
                    ${renderChunkListHtml(rerankMod.chunks || [], 'obs-rerank-list', 'reranker')}
                </div>
            </div>
            ${renderModuleTimingBar(rerankMod.timing, '未执行 Reranker 重排')}
        </div>

        ${renderDashedConnector('重排结果进入断崖检测 (绝对分差 > 0.25 或 相对分差 > 10%)', !cliffMod.executed)}

        <!-- ==================== 模块 6：断崖检测的 Chunks 模块 ==================== -->
        <div class="trace-module-wrap ${cliffMod.executed ? '' : 'is-inactive-module'}">
            <div class="trace-module-box">
                <div class="trace-module-header">
                    <div class="trace-module-title">
                        <span class="trace-step-num">6</span>
                        <span>断崖检测 Chunks (Cliff Detection · 被断崖截断的切片标记灰色)</span>
                    </div>
                    <div class="trace-module-actions">
                        <span class="obs-tag-pill item-pill">保留送入LLM: ${cliffMod.kept_count || 0} 条</span>
                        <span class="obs-tag-pill raw-item-pill">被断崖标灰: ${cliffMod.dropped_count || 0} 条</span>
                        <button class="btn-fold-toggle" onclick="toggleAllChunksInBox(this, 'obs-cliff-list')">🔽 全部展开</button>
                    </div>
                </div>
                <div class="trace-module-body">
                    <div style="margin-bottom:10px;font-size:12px;color:#786657;background:#faf7f0;padding:8px 12px;border-radius:6px;border:1px solid #e9dec4;">
                        <strong>⚙️ 断崖参数：</strong>
                        ABS_GAP_THRES = <code>${cliffMod.abs_gap_threshold ?? 0.25}</code> ·
                        RATIO_GAP_THRES = <code>${cliffMod.ratio_gap_threshold ?? 0.10}</code> ·
                        保留区间 = <code>[${cliffMod.min_len ?? 3}, ${cliffMod.max_len ?? 10}]</code> ·
                        <strong>结论：</strong>${escapeHtml(cliffMod.trigger_reason || '未执行')}
                    </div>
                    ${renderChunkListHtml(cliffMod.chunks || [], 'obs-cliff-list', 'cliff', { trigger_reason: cliffMod.trigger_reason })}
                </div>
            </div>
            ${renderModuleTimingBar(cliffMod.timing, '未执行断崖检测')}
        </div>

        ${renderDashedConnector('保留的上下文 Chunks 拼入 Prompt 送交大模型流式生成')}

        <!-- ==================== 模块 7：最终模型的输出模块 ==================== -->
        <div class="trace-module-wrap">
            <div class="trace-module-box">
                <div class="trace-module-header">
                    <div class="trace-module-title">
                        <span class="trace-step-num">7</span>
                        <span>最终模型输出 (Final Model Answer Output)</span>
                    </div>
                    <div class="trace-module-actions">
                        <span class="obs-tag-pill kw-pill">模式: ${escapeHtml(outMod.output_mode || 'rag_llm')}</span>
                        <span class="obs-tag-pill item-pill">引用角标: ${(outMod.cited_chunk_ids || []).map(i => `[${i}]`).join(' ') || '无'}</span>
                    </div>
                </div>
                <div class="trace-module-body">
                    <div class="bubble" style="max-width:100%;width:100%;background:#fffdf9;">
                        ${formatAnswerToHtml(outMod.final_answer || '（无输出内容）', trace.trace_id)}
                    </div>
                </div>
            </div>
            ${renderModuleTimingBar(outMod.timing)}
        </div>
    </div>
    `;

    obsCanvasEl.innerHTML = html;
}

if (btnRefreshTracesEl) {
    btnRefreshTracesEl.addEventListener('click', () => loadObserverFolders(true));
}

if (obsSearchInputEl) {
    obsSearchInputEl.addEventListener('input', (e) => {
        renderObserverFolderTree(e.target.value || '');
    });
}
