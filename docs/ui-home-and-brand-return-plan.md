# 音量球改动与 DiTing 返回按钮计划书

## 目标

本计划只记录两项前端改动：

- 首页音量球及其周围动态效果的优化。
- 左上角 `DiTing` 品牌区作为返回主页按钮的交互设置。

涉及文件：

- `frontend/index.html`

## 一、音量球改动

### 当前实现

首页音量球位于 `homeHero` 区域中：

```html
<canvas class="freq-ring" id="freqRing" width="620" height="620"></canvas>
<canvas class="particle-canvas" id="particleCanvas" width="620" height="620"></canvas>
<div class="volume-orb" id="volumeOrb"></div>
```

相关样式：

- `.volume-orb`：中心白色半透明音量球。
- `.freq-ring`：音量频谱环画布。
- `.particle-canvas`：粒子特效画布。

### 已做调整

1. 扩大频谱环画布

原画布为 `520x520`，高音量时最大半径会超出画布并被裁切。现已改为：

```html
width="620" height="620"
```

对应 CSS：

```css
.freq-ring,
.particle-canvas {
  width: 620px;
  height: 620px;
}
```

2. 限制最大音量环半径

相关 JS 位于 `startVolumeMeter()`：

```js
const cx = 310, cy = 310;
const innerR = 110, outerR = 240;
```

这样最大半径为 `240`，中心坐标为 `310`，不会超出 `620x620` 画布范围。

3. 增加粒子效果

新增 `particleCanvas`，在音量球周围绘制蓝紫色粒子。

粒子行为：

- 默认围绕音量球缓慢漂浮。
- 音量越大，粒子生成距离越远。
- 音量越大，粒子向上漂移速度略快。
- 粒子生命周期结束后自动重生。

核心参数：

```js
const MAX_PARTICLES = 60;
const orbR = 100;
```

### 验收标准

- 首页音量球正常显示。
- 频谱环在高音量时不被裁切。
- 粒子围绕音量球持续运动。
- 音量变化时粒子扩散强度有变化。
- 不遮挡主标题和副标题。

## 二、DiTing 返回主页按钮

### 当前实现

顶部左上角品牌区：

```html
<div class="topbar-brand" id="topbarBrand" title="返回主页">
  <div class="topbar-brand-mark">D</div>
  <span>DiTing</span>
</div>
```

初始化时绑定点击事件：

```js
const topbarBrand = $('topbarBrand');
if (topbarBrand) topbarBrand.addEventListener('click', showHome);
```

样式中应保留：

```css
.topbar-brand {
  cursor: pointer;
  user-select: none;
}
```

### 交互目标

点击左上角 `DiTing` 后返回首页。

应覆盖以下场景：

- 从会议转写返回主页。
- 从会议预约返回主页。
- 从会议分析返回主页。
- 从用户管理返回主页。

### 关键逻辑

返回主页主要调用：

```js
showHome();
```

需要确保 `showHome()` 做到：

- 隐藏模块页面。
- 显示首页 Hero。
- 清空当前模块状态。
- 更新顶部导航状态。
- 恢复或启动首页音量球动画。
- 清理管理面板残留样式。

建议 `showHome()` 中保留以下清理逻辑：

```js
if (managementPanel) {
  managementPanel.classList.remove('visible');
  managementPanel.removeAttribute('style');
}
```

原因：预约、分析、用户管理模块会给 `managementPanel` 写入全屏内联样式，如果返回主页时不清理，可能遮挡首页。

### 验收标准

- 鼠标移到左上角品牌区显示可点击状态。
- 点击 `DiTing` 后立即回到首页。
- 返回后管理面板不残留、不遮挡。
- 顶部导航不再错误高亮某个模块。
- 不触发重新登录或多余 toast。

## 注意事项

- `frontend/index.html` 使用 `<script type="module">`，模块内函数不会自动挂到 `window`。
- 不要使用 `onclick="showHome()"` 这种内联写法，应通过 `addEventListener` 绑定。
- 修改音量球时必须同时检查 HTML canvas 尺寸、CSS 尺寸和 JS 清屏尺寸是否一致。
