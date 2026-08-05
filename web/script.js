document.addEventListener('DOMContentLoaded', function() {
    // 元素引用
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const processBtn = document.getElementById('processBtn');
    const loading = document.getElementById('loading');
    const outputPlaceholder = document.getElementById('outputPlaceholder');
    const outputImage = document.getElementById('outputImage');
    const augEvalPlaceholder = document.getElementById('augEvalPlaceholder');
    const augEvalGrid = document.getElementById('augEvalGrid');

    // 当前上传的图像
    let currentImage = null;
    let currentImageUrl = null;

    // 最近一次增广结果缓存（用于保存等操作）
    let lastAugmentedResults = [];
    let lastBestResultIndex = 0;

    // 上传区域点击事件
    uploadArea.addEventListener('click', function(e) {
        if (e.target !== fileInput) {
            fileInput.click();
        }
    });

    // 拖拽上传支持
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.style.borderColor = '#2980b9';
        uploadArea.style.background = '#d6eaf8';
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.style.borderColor = '#3498db';
        uploadArea.style.background = '#e8f4fc';
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.style.borderColor = '#3498db';
        uploadArea.style.background = '#e8f4fc';

        if (e.dataTransfer.files.length) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    // 文件选择事件
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // 处理文件选择
    function handleFileSelect(file) {
        // 检查文件类型
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        if (!validTypes.includes(file.type)) {
            alert('请上传JPG或PNG格式的图片');
            return;
        }

        // 检查文件大小（5MB）
        if (file.size > 5 * 1024 * 1024) {
            alert('图片大小不能超过5MB');
            return;
        }

        // 清除之前的预览
        if (currentImageUrl) {
            URL.revokeObjectURL(currentImageUrl);
        }

        // 创建预览
        currentImageUrl = URL.createObjectURL(file);
        currentImage = file;

        imagePreview.innerHTML = `
            <div>
                <img src="${currentImageUrl}" alt="上传的图片">
                <p class="file-info">${file.name} (${(file.size / 1024).toFixed(1)} KB)</p>
            </div>
        `;

        // 启用处理按钮
        processBtn.disabled = false;

        // 重置输出区域
        outputPlaceholder.style.display = 'block';
        outputImage.innerHTML = '';
        resetAugEvaluation();
        lastAugmentedResults = [];
        lastBestResultIndex = 0;

        // 隐藏增广图像展示区
        const augSection = document.getElementById('augmentedResultsSection');
        if (augSection) augSection.style.display = 'none';
    }

    // 处理按钮点击事件
    processBtn.addEventListener('click', function() {
        if (!currentImage) {
            alert('请先上传一张图片');
            return;
        }

        // 显示加载动画
        loading.style.display = 'block';
        processBtn.disabled = true;
        outputPlaceholder.style.display = 'none';

        // 尝试调用API处理图像
        processWithAPI(currentImage).then(result => {
            // API调用成功
            displayResults(result);
        }).catch(error => {
            console.warn('API调用失败，使用模拟数据:', error);
            // API调用失败，使用模拟数据
            setTimeout(() => {
                simulateGeneration();
            }, 1000);
        }).finally(() => {
            // 隐藏加载动画
            loading.style.display = 'none';
            processBtn.disabled = false;
        });
    });

    // 使用API处理图像
    async function processWithAPI(imageFile) {
        const formData = new FormData();
        formData.append('image', imageFile);
        const category = document.getElementById('categorySelect').value;
        if (category) {
            formData.append('category', category);
        }

        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`API错误: ${response.status}`);
            }

            const result = await response.json();

            if (!result.success) {
                throw new Error(result.error || '处理失败');
            }

            return result;
        } catch (error) {
            console.error('API调用失败:', error);
            throw error;
        }
    }


    // 全屏查看
    let fullscreenImageDataUrl = null;
    function openFullscreen(dataUrl) {
        fullscreenImageDataUrl = dataUrl;
        const modal = document.getElementById('fullscreenModal');
        const img = document.getElementById('fullscreenImage');
        img.src = dataUrl;
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    function closeFullscreen() {
        const modal = document.getElementById('fullscreenModal');
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }

    // 手动选择最佳结果（保留点击高亮逻辑供切换评估使用）
    function selectAsBest(index) {
        lastBestResultIndex = index;
        // 更新所有 aug-item 的样式
        const augItems = document.querySelectorAll('.aug-item');
        augItems.forEach((item, i) => {
            const badge = item.querySelector('.best-result-badge');
            const selectBtn = item.querySelector('.select-best-btn');
            if (i === index) {
                item.classList.add('best-result');
                if (badge) badge.style.display = '';
                if (selectBtn) {
                    selectBtn.classList.add('active');
                    selectBtn.innerHTML = '<i class="fas fa-star"></i> 已选为最佳';
                }
            } else {
                item.classList.remove('best-result');
                if (badge) badge.style.display = 'none';
                if (selectBtn) {
                    selectBtn.classList.remove('active');
                    selectBtn.innerHTML = '<i class="far fa-star"></i> 选为最佳';
                }
            }
        });
        // 更新最佳摘要
        updateBestSummary(index + 1);
        showToast(`已将增广结果 ${index + 1} 标记为最佳`);
    }

    // 更新最佳结果摘要
    function updateBestSummary(bestNumber) {
        const summaryEl = document.getElementById('bestSummary');
        if (summaryEl) {
            summaryEl.innerHTML = `
                <h4><i class="fas fa-trophy"></i> 当前最佳增广结果：第 ${bestNumber} 张</h4>
                <p>该结果在缺陷真实性和背景保持之间取得了最佳平衡，建议优先使用该图像进行后续训练。</p>
            `;
        }
    }

    // 显示处理结果
    function displayResults(result) {
        const augmentedResults = Array.isArray(result.augmented_results) && result.augmented_results.length
            ? result.augmented_results
            : [result.output_image];

        // 缓存结果
        lastAugmentedResults = augmentedResults;
        lastBestResultIndex = 0;

        // ===== 1. 输出图像栏 =====
        const fallbackReason = result.fallback_reason || '';
        const fallbackBanner = result.mode === 'fallback_simulation'
            ? `<div class="fallback-mode-banner"><i class="fas fa-exclamation-triangle"></i> 当前为<strong>后备模拟</strong>（未成功调用训练权重或推理异常），下图仅为示意叠加，不是模型真实增广。${fallbackReason ? `<br><small style="color:#935116">原因: ${fallbackReason}</small>` : ''}</div>`
            : '';

        outputImage.innerHTML = `
            <div class="output-display">
                ${fallbackBanner}
                <img src="${result.output_image}" alt="生成图像" id="generatedOutputImg">
                <div class="output-actions">
                    <button class="output-action-btn fullscreen-btn" onclick="window._openFullscreen('${result.output_image}')" title="全屏查看">
                        <i class="fas fa-expand"></i> 全屏
                    </button>
                </div>
            </div>
        `;

        // ===== 2. 增广图像展示区（4张，点击切换评估） =====
        const augSection = document.getElementById('augmentedResultsSection');
        const augGrid = document.getElementById('augmentedGrid');

        if (augSection && augGrid) {
            augSection.style.display = 'block';

            const labels = Array.isArray(result.augmented_labels) ? result.augmented_labels : [];
            // 每张增广图像的独立分析（如果有）
            const perAnalyses = Array.isArray(result.augmented_analyses) ? result.augmented_analyses : [];

            const augmentedGridHtml = augmentedResults.map((imgSrc, index) => {
                const label = labels[index] || `增广结果 ${index + 1}`;
                const selectedClass = index === 0 ? ' selected' : '';
                return `
                <div class="aug-item${selectedClass}" data-aug-index="${index}" onclick="window._onAugItemClick(${index})">
                    <img src="${imgSrc}" alt="${label}">
                    <p>${label}</p>
                </div>
            `;}).join('');

            augGrid.innerHTML = augmentedGridHtml;

            // 缓存每张增广的分析数据
            window._augAnalyses = perAnalyses.length === augmentedResults.length ? perAnalyses : null;
            window._augInputB64 = result.input_image;

            // 默认展示第一张的评估
            if (perAnalyses.length > 0) {
                renderAugEvaluation(perAnalyses[0], 0);
            } else {
                renderAugEvaluation(result.analysis, 0);
            }
        }

        // 显示主评估摘要（用主输出的分析）
        renderAugEvaluation(result.analysis, -1);  // -1 表示主输出
    }

    // 点击增广图像切换评估
    window._onAugItemClick = function(index) {
        // 更新选中样式
        document.querySelectorAll('.aug-item').forEach((item, i) => {
            if (i === index) {
                item.classList.add('selected');
            } else {
                item.classList.remove('selected');
            }
        });

        // 更新评估面板
        if (window._augAnalyses && window._augAnalyses[index]) {
            renderAugEvaluation(window._augAnalyses[index], index);
        }
        lastBestResultIndex = index;
    };

    function resetAugEvaluation() {
        if (augEvalPlaceholder) augEvalPlaceholder.style.display = 'block';
        if (augEvalGrid) {
            augEvalGrid.style.display = 'none';
            augEvalGrid.innerHTML = '';
        }
    }

    function renderAugEvaluation(analysis, augIndex) {
        if (!analysis) {
            if (augEvalPlaceholder) augEvalPlaceholder.style.display = 'block';
            if (augEvalGrid) augEvalGrid.style.display = 'none';
            return;
        }

        const evalItems = [
            { label: '缺陷类型', value: analysis.defect_type || '—' },
            { label: '异常显著度', value: `${analysis.anomaly_salience || 0}%` },
            { label: '缺陷位置', value: analysis.defect_location || '—' },
            { label: '生成质量', value: analysis.generation_quality || '—' },
            { label: 'FID', value: analysis.fid_score != null ? analysis.fid_score : '—' },
            { label: 'IS', value: analysis.is_score != null ? analysis.is_score : '—' },
            { label: 'LPIPS', value: analysis.lpips_score != null ? analysis.lpips_score : '—' },
            { label: 'PSNR', value: analysis.psnr_score != null ? analysis.psnr_score : '—' },
        ];

        if (augEvalGrid) {
            augEvalGrid.innerHTML = evalItems.map(item => `
                <div class="aug-eval-item">
                    <div class="eval-label">${item.label}</div>
                    <div class="eval-value">${item.value}</div>
                </div>
            `).join('');
            augEvalGrid.style.display = 'grid';
        }
        if (augEvalPlaceholder) augEvalPlaceholder.style.display = 'none';
    }

    // 模拟图像生成
    async function simulateGeneration() {
        const noise1 = await createFallbackPreview(currentImageUrl, 1);
        const noise2 = await createFallbackPreview(currentImageUrl, 2);
        const noise3 = await createFallbackPreview(currentImageUrl, 3);
        const patch1 = await createPatchCompositePreview(currentImageUrl, 10);
        const patch2 = await createPatchCompositePreview(currentImageUrl, 20);
        const rnd = (a, b) => Math.round((a + Math.random() * (b - a)) * 10) / 10;
        const mockResult = {
            success: true,
            mode: 'fallback_simulation',
            input_image: currentImageUrl,
            output_image: noise2,
            augmented_results: [noise1, noise2, noise3, patch1, patch2],
            augmented_labels: [
                '模拟 · 划痕风格（轻微伪异常）',
                '模拟 · 斑点风格（标准异常）',
                '模拟 · 纹理风格（区域性异常）',
                '模拟 · 贴片风格 A（局部缺陷）',
                '模拟 · 贴片风格 B（局部缺陷）',
            ],
            analysis: {
                anomaly_salience: Math.round(78 + Math.random() * 18),
                defect_location: '右上区域',
                defect_type: '划痕',
                generation_quality: '高',
                fid_score: rnd(16, 22),
                is_score: rnd(2.8, 3.6),
                lpips_score: rnd(0.26, 0.38),
                psnr_score: rnd(26, 31)
            },
            message: '未连接到后端服务，当前为浏览器本地模拟结果。请用 Flask 启动 web/app.py 后再试。'
        };

        displayResults(mockResult);
    }

    // 更真实的工业缺陷模拟（后备方案）
    function createFallbackPreview(baseImageUrl, seed = 0) {
        return new Promise((resolve) => {
            const canvas = document.createElement('canvas');
            const size = 512;
            canvas.width = size;
            canvas.height = size;
            const ctx = canvas.getContext('2d');

            const img = new Image();
            img.onload = function() {
                // 绘制原图
                ctx.drawImage(img, 0, 0, size, size);

                // 获取图像数据用于像素级操作
                const imageData = ctx.getImageData(0, 0, size, size);
                const pixels = imageData.data;

                // 简易伪随机（基于seed）
                const rng = (s) => { const x = Math.sin(s * 127.1 + seed * 31.7) * 43758.5453; return x - Math.floor(x); };

                // 简易2D噪声
                function noise2D(x, y) {
                    const n = Math.sin(x * 12.9898 + y * 78.233 + seed * 43.1) * 43758.5453;
                    return (n - Math.floor(n)) * 2 - 1;
                }

                // 平滑噪声
                function smoothNoise(x, y, scale) {
                    const sx = x / scale, sy = y / scale;
                    const ix = Math.floor(sx), iy = Math.floor(sy);
                    const fx = sx - ix, fy = sy - iy;
                    const sx1 = fx * fx * (3 - 2 * fx);
                    const sy1 = fy * fy * (3 - 2 * fy);
                    const n00 = noise2D(ix, iy);
                    const n10 = noise2D(ix + 1, iy);
                    const n01 = noise2D(ix, iy + 1);
                    const n11 = noise2D(ix + 1, iy + 1);
                    return n00 * (1 - sx1) * (1 - sy1) + n10 * sx1 * (1 - sy1) + n01 * (1 - sx1) * sy1 + n11 * sx1 * sy1;
                }

                // 分形噪声
                function fbm(x, y) {
                    let val = 0, amp = 1, freq = 1, total = 0;
                    for (let i = 0; i < 4; i++) {
                        val += smoothNoise(x * freq, y * freq, 8) * amp;
                        total += amp;
                        amp *= 0.5;
                        freq *= 2;
                    }
                    return val / total;
                }

                const defectType = Math.floor(rng(1) * 3); // 0: 划痕, 1: 斑点/污渍, 2: 纹理异常

                for (let y = 0; y < size; y++) {
                    for (let x = 0; x < size; x++) {
                        const idx = (y * size + x) * 4;
                        let defectMask = 0; // 0 = 无缺陷, 1 = 完全缺陷

                        const cx = size / 2, cy = size / 2;

                        if (defectType === 0) {
                            // 划痕：多条不规则曲线
                            const sx = x - cx, sy = y - cy;
                            const distFromCenter = Math.sqrt(sx * sx + sy * sy) / (size * 0.45);
                            const angle = Math.atan2(sy + fbm(x, y + 5) * 12, sx + fbm(x + 10, y) * 12);

                            let scratchVal = 0;
                            for (let s = 0; s < 3; s++) {
                                const sa = (s * 2.1 + seed * 0.7) % (Math.PI * 2);
                                const sw = 3.5 + rng(s * 7 + seed) * 5;
                                const sr = 28 + rng(s * 11 + seed) * 55;
                                const sox = Math.cos(sa) * sr;
                                const soy = Math.sin(sa) * sr;
                                const sdx = x - (cx + sox), sdy = y - (cy + soy);
                                const srotX = sdx * Math.cos(-sa) - sdy * Math.sin(-sa);
                                const srotY = sdx * Math.sin(-sa) + sdy * Math.cos(-sa);
                                const ndist = Math.abs(srotY + fbm(x * 0.3, y * 0.3) * 3);
                                const alongDist = Math.abs(srotX) / 35;
                                if (ndist < sw && alongDist < 1) {
                                    scratchVal = Math.max(scratchVal, (1 - ndist / sw) * (1 - alongDist) * 0.85);
                                }
                            }
                            defectMask = scratchVal;
                        } else if (defectType === 1) {
                            // 斑点/污渍：不规则的暗色区域
                            const spots = [
                                { ox: cx + 10 + rng(2) * 40, oy: cy - 5 + rng(3) * 30, rx: 18 + rng(4) * 22, ry: 12 + rng(5) * 18 },
                                { ox: cx - 15 + rng(6) * 35, oy: cy + 8 + rng(7) * 28, rx: 14 + rng(8) * 16, ry: 10 + rng(9) * 14 },
                            ];
                            let spotVal = 0;
                            spots.forEach(sp => {
                                const dx = (x - sp.ox) / sp.rx, dy = (y - sp.oy) / sp.ry;
                                const ed = dx * dx + dy * dy;
                                const n = fbm(x * 0.4, y * 0.4) * 0.3;
                                const spotStrength = Math.exp(-ed * 2.5) * (0.75 + n);
                                spotVal = Math.max(spotVal, spotStrength);
                            });
                            // 边缘不规则
                            const edgeNoise = fbm(x * 0.6, y * 0.6) * 0.2;
                            defectMask = Math.min(1, spotVal + edgeNoise);
                        } else {
                            // 纹理异常：局部纹理变化
                            const dx = x - (cx - 8 + rng(2) * 50), dy = y - (cy + 5 + rng(3) * 40);
                            const dist = Math.sqrt(dx * dx + dy * dy);
                            const radius = 50 + rng(4) * 30;
                            if (dist < radius) {
                                const texNoise = (fbm(x * 0.5, y * 0.5) * 2 - 1) * 0.6;
                                const falloff = 1 - Math.pow(dist / radius, 2);
                                defectMask = Math.max(0, texNoise * falloff);
                            }
                        }

                        if (defectMask > 0.01) {
                            // 缺陷区域仅轻微变暗，保持原图色调
                            const darken = defectMask * 0.35;
                            pixels[idx]     = Math.max(0, pixels[idx]     * (1 - darken));
                            pixels[idx + 1] = Math.max(0, pixels[idx + 1] * (1 - darken));
                            pixels[idx + 2] = Math.max(0, pixels[idx + 2] * (1 - darken));
                        }
                    }
                }

                ctx.putImageData(imageData, 0, 0);
                resolve(canvas.toDataURL('image/jpeg', 0.95));
            };
            img.onerror = function() {
                resolve(baseImageUrl);
            };
            img.src = baseImageUrl;
        });
    }

    // 废弃旧的简单覆盖（保留兼容引用）
    function createFallbackPreviewOld(baseImageUrl, offsetX, offsetY) {
        return createFallbackPreview(baseImageUrl, offsetX);
    }

    // 基于缺陷区域堆叠的增广（模拟缺陷样本库合成）
    function createPatchCompositePreview(baseImageUrl, seed = 0) {
        return new Promise((resolve) => {
            const canvas = document.createElement('canvas');
            const size = 512;
            canvas.width = size;
            canvas.height = size;
            const ctx = canvas.getContext('2d');

            const img = new Image();
            img.onload = function() {
                // 绘制原图
                ctx.drawImage(img, 0, 0, size, size);

                const rng = (s) => { const x = Math.sin(s * 127.1 + seed * 31.7) * 43758.5453; return x - Math.floor(x); };

                // 从原图随机区域提取"缺陷块"
                const patchW = 22 + Math.floor(rng(1) * 20);
                const patchH = 16 + Math.floor(rng(2) * 16);
                const srcX = Math.floor(rng(3) * (size - patchW));
                const srcY = Math.floor(rng(4) * (size - patchH));
                const srcPatch = ctx.getImageData(srcX, srcY, patchW, patchH);

                // 处理缺陷块：变暗 + 添加纹理噪声
                const sp = srcPatch.data;
                for (let i = 0; i < sp.length; i += 4) {
                    const n = (Math.sin(i * 0.07 + seed * 5.3) * 0.5 + 0.5);
                    const dark = 0.25 + n * 0.25;
                    sp[i]     = Math.max(0, sp[i]     * (1 - dark));
                    sp[i + 1] = Math.max(0, sp[i + 1] * (1 - dark));
                    sp[i + 2] = Math.max(0, sp[i + 2] * (1 - dark));
                }

                // 将处理后的缺陷块贴到原图不同位置
                const dstX = Math.floor(rng(5) * (size - patchW));
                const dstY = Math.floor(rng(6) * (size - patchH));

                // 创建临时canvas做alpha混合
                const tmpCanvas = document.createElement('canvas');
                tmpCanvas.width = patchW;
                tmpCanvas.height = patchH;
                const tmpCtx = tmpCanvas.getContext('2d');
                tmpCtx.putImageData(srcPatch, 0, 0);

                // 用羽化遮罩实现边缘渐变融合
                const maskCanvas = document.createElement('canvas');
                maskCanvas.width = patchW;
                maskCanvas.height = patchH;
                const maskCtx = maskCanvas.getContext('2d');
                const gradient = maskCtx.createRadialGradient(patchW/2, patchH/2, patchW*0.15, patchW/2, patchH/2, patchW*0.75);
                gradient.addColorStop(0, 'rgba(0,0,0,0.9)');
                gradient.addColorStop(0.6, 'rgba(0,0,0,0.5)');
                gradient.addColorStop(1, 'rgba(0,0,0,0)');
                maskCtx.fillStyle = gradient;
                maskCtx.fillRect(0, 0, patchW, patchH);

                // 合成：用遮罩控制透明度
                const maskData = maskCtx.getImageData(0, 0, patchW, patchH).data;
                const dstImageData = ctx.getImageData(dstX, dstY, patchW, patchH);
                const dp = dstImageData.data;
                for (let i = 0; i < dp.length; i += 4) {
                    const alpha = maskData[i] / 255; // 使用mask的R通道作为alpha
                    dp[i]     = dp[i]     * (1 - alpha) + sp[i]     * alpha;
                    dp[i + 1] = dp[i + 1] * (1 - alpha) + sp[i + 1] * alpha;
                    dp[i + 2] = dp[i + 2] * (1 - alpha) + sp[i + 2] * alpha;
                }
                ctx.putImageData(dstImageData, dstX, dstY);

                // 再贴第二个缺陷块
                const patchW2 = 16 + Math.floor(rng(7) * 14);
                const patchH2 = 12 + Math.floor(rng(8) * 12);
                const srcX2 = Math.floor(rng(9) * (size - patchW2));
                const srcY2 = Math.floor(rng(10) * (size - patchH2));
                const srcPatch2 = ctx.getImageData(srcX2, srcY2, patchW2, patchH2);
                const sp2 = srcPatch2.data;
                for (let i = 0; i < sp2.length; i += 4) {
                    const n = (Math.sin(i * 0.09 + seed * 7.1) * 0.5 + 0.5);
                    const dark = 0.2 + n * 0.3;
                    sp2[i]     = Math.max(0, sp2[i]     * (1 - dark));
                    sp2[i + 1] = Math.max(0, sp2[i + 1] * (1 - dark));
                    sp2[i + 2] = Math.max(0, sp2[i + 2] * (1 - dark));
                }

                const dstX2 = Math.floor(rng(11) * (size - patchW2));
                const dstY2 = Math.floor(rng(12) * (size - patchH2));
                const maskCanvas2 = document.createElement('canvas');
                maskCanvas2.width = patchW2;
                maskCanvas2.height = patchH2;
                const maskCtx2 = maskCanvas2.getContext('2d');
                const gradient2 = maskCtx2.createRadialGradient(patchW2/2, patchH2/2, patchW2*0.1, patchW2/2, patchH2/2, patchW2*0.7);
                gradient2.addColorStop(0, 'rgba(0,0,0,0.85)');
                gradient2.addColorStop(0.7, 'rgba(0,0,0,0.3)');
                gradient2.addColorStop(1, 'rgba(0,0,0,0)');
                maskCtx2.fillStyle = gradient2;
                maskCtx2.fillRect(0, 0, patchW2, patchH2);
                const maskData2 = maskCtx2.getImageData(0, 0, patchW2, patchH2).data;

                const dstImageData2 = ctx.getImageData(dstX2, dstY2, patchW2, patchH2);
                const dp2 = dstImageData2.data;
                for (let i = 0; i < dp2.length; i += 4) {
                    const alpha = maskData2[i] / 255;
                    dp2[i]     = dp2[i]     * (1 - alpha) + sp2[i]     * alpha;
                    dp2[i + 1] = dp2[i + 1] * (1 - alpha) + sp2[i + 1] * alpha;
                    dp2[i + 2] = dp2[i + 2] * (1 - alpha) + sp2[i + 2] * alpha;
                }
                ctx.putImageData(dstImageData2, dstX2, dstY2);

                resolve(canvas.toDataURL('image/jpeg', 0.95));
            };
            img.onerror = function() {
                resolve(baseImageUrl);
            };
            img.src = baseImageUrl;
        });
    }

    // Toast 通知
    function showToast(message) {
        const existing = document.querySelector('.save-toast');
        if (existing) existing.remove();

        const toast = document.createElement('div');
        toast.className = 'save-toast';
        toast.textContent = message;
        document.body.appendChild(toast);

        setTimeout(() => {
            if (toast.parentNode) toast.parentNode.removeChild(toast);
        }, 2500);
    }

    // 暴露函数到全局作用域
    window._openFullscreen = openFullscreen;
    window._closeFullscreen = closeFullscreen;
    window._selectAsBest = selectAsBest;

    // 全屏模态框事件绑定
    const fullscreenModal = document.getElementById('fullscreenModal');
    const fullscreenBackdrop = document.getElementById('fullscreenBackdrop');
    const fullscreenClose = document.getElementById('fullscreenClose');

    if (fullscreenBackdrop) {
        fullscreenBackdrop.addEventListener('click', closeFullscreen);
    }
    if (fullscreenClose) {
        fullscreenClose.addEventListener('click', closeFullscreen);
    }

    // ESC 键关闭全屏
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && fullscreenModal && fullscreenModal.classList.contains('active')) {
            closeFullscreen();
        }
    });

    // ---- 加载产品类别列表 ----
    async function loadCategories() {
        try {
            const resp = await fetch('/api/categories');
            const data = await resp.json();
            if (data.success && data.categories) {
                const select = document.getElementById('categorySelect');
                const info = document.getElementById('categoryInfo');
                data.categories.forEach(cat => {
                    const opt = document.createElement('option');
                    opt.value = cat.name;
                    opt.textContent = `${cat.name} (${cat.defect_count}种缺陷)`;
                    opt.dataset.defects = cat.defect_types.join(', ');
                    select.appendChild(opt);
                });
                select.addEventListener('change', function() {
                    const sel = this.options[this.selectedIndex];
                    if (sel.dataset.defects) {
                        info.textContent = '缺陷类型: ' + sel.dataset.defects;
                    } else {
                        info.textContent = '';
                    }
                });
            }
        } catch (e) {
            console.warn('无法加载产品类别列表:', e);
        }
    }
    loadCategories();

    // 初始化
    resetAugEvaluation();

    // 初始禁用处理按钮
    processBtn.disabled = true;
});
