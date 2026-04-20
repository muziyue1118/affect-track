import pygame
import math
import sys


class ComponentBase:
    def __init__(self, x=0., y=0., w=0., h=0., center=(0., 0.)):
        """
        初始化组件
        :param x: 相对x坐标 (0-1)
        :param y: 相对y坐标 (0-1)
        :param w: 相对宽度 (0-1)
        :param h: 相对高度 (0-1)
        :param center: 原点位置 (0-1, 0-1)
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.center = center

    def get_coord(self, screen_width, screen_height):
        """
        获取绝对坐标和尺寸
        :param screen_width: 屏幕宽度
        :param screen_height: 屏幕高度
        :return: (x, y, w, h) 绝对坐标和尺寸
        """
        # 计算宽度和高度
        abs_w = int(self.w * screen_width)
        abs_h = int(self.h * screen_height)

        # 计算原点偏移
        origin_x = self.center[0] * abs_w
        origin_y = self.center[1] * abs_h

        # 计算绝对坐标
        abs_x = int(self.x * screen_width) - origin_x
        abs_y = int(self.y * screen_height) - origin_y

        return abs_x, abs_y, abs_w, abs_h

    def proj(self, x, y, screen_width, screen_height):
        """
        将相对坐标转换为绝对坐标
        :param x: 相对x坐标 (0-1)
        :param y: 相对y坐标 (0-1)
        :param screen_width: 屏幕宽度
        :param screen_height: 屏幕高度
        :return: (x, y) 绝对坐标
        """
        abs_x = int(x * screen_width)
        abs_y = int(y * screen_height)
        return abs_x, abs_y


class App:
    def __init__(self, screen_width, screen_height, caption):
        pygame.init()
        screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
        pygame.display.set_caption(caption)
        pygame.mixer.init()
        self.screen = screen
        self.clock = pygame.time.Clock()

    def quit(self):
        pygame.quit()

class SceneBase:
    def __init__(self, app:App):
        self.app = app
        self.running = True
        self.tick_offset = 120

    def event_parse(self, event):
        if event.type == pygame.QUIT:
            self.app.quit()
            self.running = False
        if event.type == pygame.VIDEORESIZE:
            # 窗口大小改变时自动调整
            screen_width, screen_height = event.size
            self.app.screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)

    def listen(self):
        for event in pygame.event.get():
            self.event_parse(event)

    def draw(self):
        pass

    def enter(self):
        while self.running:
            self.listen()
            self.draw()
            pygame.display.flip()
            self.app.clock.tick(self.tick_offset)
        return self

class Rect(ComponentBase):
    def __init__(self, x=0, y=0, w=0, h=0, center=(0, 0), color=(255, 255, 255)):
        super().__init__(x, y, w, h, center)
        self.color = color

    def draw(self, screen):
        screen_width, screen_height = screen.get_size()
        x, y, w, h = self.get_coord(screen_width, screen_height)
        pygame.draw.rect(screen, self.color, (x, y, w, h))

class MultiText(ComponentBase):
    def __init__(self, x=0., y=0., w=0., h=0., center=(0.5, 0.5),
                 text="", color=(255, 255, 255), bg_color=None,
                 font_size=24, font_name="simsun", line_spacing=1.2,
                 align="center"):
        super().__init__(x, y, w, h, center)
        self.text = text
        self.color = color
        self.bg_color = bg_color
        self.font_size = font_size  # 基准字体大小
        self.font_name = font_name
        self.line_spacing = line_spacing
        self.align = align.lower()
        self._font = None
        self._last_size = (0, 0)  # 记录上次的屏幕大小
        self._cached_surface = None  # 缓存渲染结果

    def _update_font(self, screen_width, screen_height):
        """根据窗口大小动态调整字体"""
        # 计算缩放后的字体大小（基于屏幕高度）
        scaled_size = max(12, int(self.font_size * min(screen_width, screen_height) / 800))

        # 当屏幕大小变化或字体未初始化时创建新字体
        if (screen_width, screen_height) != self._last_size or self._font is None:
            try:
                self._font = pygame.font.SysFont(self.font_name, scaled_size)
                self._last_size = (screen_width, screen_height)
                self._cached_surface = None  # 清除缓存
            except:
                self._font = pygame.font.SysFont(None, scaled_size)

    def _render_multiline_text(self):
        """渲染带对齐的多行文本"""
        lines = self.text.split('\n')
        surfaces = []
        max_width = 0

        # 渲染每行文本
        for line in lines:
            if line:
                surf = self._font.render(line, True, self.color)
                surfaces.append(surf)
                max_width = max(max_width, surf.get_width())
            else:
                surfaces.append(None)  # 保留空行间距

        # 计算总高度（考虑行间距）
        line_height = self._font.get_linesize()
        total_height = line_height * (len(lines) * self.line_spacing - (self.line_spacing - 1))

        # 创建目标Surface
        combined = pygame.Surface((max_width, total_height), pygame.SRCALPHA)
        y_offset = 0

        # 合并所有行（处理对齐）
        for surf in surfaces:
            if surf:
                if self.align == "left":
                    x_pos = 0
                elif self.align == "right":
                    x_pos = max_width - surf.get_width()
                else:  # center
                    x_pos = (max_width - surf.get_width()) // 2

                combined.blit(surf, (x_pos, y_offset))
            y_offset += line_height * self.line_spacing

        return combined

    def draw(self, screen):
        screen_width, screen_height = screen.get_size()

        # 更新字体（窗口大小变化时会自动调整）
        self._update_font(screen_width, screen_height)

        # 获取组件绝对坐标
        abs_x, abs_y, abs_w, abs_h = self.get_coord(screen_width, screen_height)

        # 重新渲染文本（如果内容或字体有变化）
        if self._cached_surface is None:
            self._cached_surface = self._render_multiline_text()

        # 计算居中位置
        text_rect = self._cached_surface.get_rect()
        text_x = abs_x + (abs_w - text_rect.width) * self.center[0]
        text_y = abs_y + (abs_h - text_rect.height) * self.center[1]

        # 绘制背景
        if self.bg_color:
            pygame.draw.rect(screen, self.bg_color, (abs_x, abs_y, abs_w, abs_h))

        # 绘制文本（坐标取整避免模糊）
        screen.blit(self._cached_surface, (round(text_x), round(text_y)))

    def set_text(self, new_text):
        """动态更新文本内容"""
        if self.text != new_text:
            self.text = new_text
            self._cached_surface = None  # 清除缓存强制重绘

class Text(ComponentBase):
    def __init__(self, x=0., y=0., w=0., h=0., center=(0.5, 0.5),
                 text="", color=(255, 255, 255), bg_color=None,
                 font_size=24, font_name="simsun"):
        super().__init__(x, y, w, h, center)
        self.text = text
        self.color = color
        self.bg_color = bg_color
        self.font_size = font_size
        self.font_name = font_name
        self._font = None
        self._last_size = (0, 0)

    def _update_font(self, screen_width, screen_height):
        """根据屏幕大小更新字体"""
        scaled_size = max(12, int(self.font_size * screen_height / 800))
        if (screen_width, screen_height) != self._last_size or self._font is None:
            try:
                self._font = pygame.font.SysFont(self.font_name, scaled_size)
                self._last_size = (screen_width, screen_height)
            except:
                self._font = pygame.font.SysFont(None, scaled_size)

    def draw(self, screen):
        screen_width, screen_height = screen.get_size()
        self._update_font(screen_width, screen_height)

        # 获取组件绝对坐标和尺寸
        abs_x, abs_y, abs_w, abs_h = self.get_coord(screen_width, screen_height)

        # 渲染文本
        text_surface = self._font.render(self.text, True, self.color)
        text_rect = text_surface.get_rect()

        # 关键修正：根据center参数计算文本位置
        text_rect.x = abs_x + (abs_w - text_rect.width) * self.center[0]
        text_rect.y = abs_y + (abs_h - text_rect.height) * self.center[1]

        # 绘制背景（如果有）
        if self.bg_color:
            pygame.draw.rect(screen, self.bg_color, (abs_x, abs_y, abs_w, abs_h))

        # 绘制文本
        screen.blit(text_surface, text_rect)


class ProgressBar(ComponentBase):
    def __init__(self, x=0, y=0, w=0, h=0, center=(0.5, 0.5),
                 border_color=(255, 255, 255), fill_color=(255, 255, 255)):
        super().__init__(x, y, w, h, center)
        self.border_color = border_color  # 边框颜色
        self.fill_color = fill_color  # 填充颜色
        self.progress = 0.0  # 进度值 (0.0-1.0)

    def draw(self, screen):
        screen_width, screen_height = screen.get_size()
        abs_x, abs_y, abs_w, abs_h = self.get_coord(screen_width, screen_height)

        # 绘制边框
        pygame.draw.rect(screen, self.border_color,
                         (abs_x, abs_y, abs_w, abs_h), 2)

        # 计算填充宽度
        fill_width = max(0, min(abs_w - 4, int((abs_w - 4) * self.progress)))

        # 绘制填充部分（留2像素边距）
        if fill_width > 0:
            pygame.draw.rect(screen, self.fill_color,
                             (abs_x + 2, abs_y + 2, fill_width, abs_h - 4))

    def set_progress(self, progress):
        """设置进度 (0.0-1.0)"""
        self.progress = max(0.0, min(1.0, progress))


class WaitScene(SceneBase):
    def __init__(self, app: App, msg: str):
        super().__init__(app)
        self.msg = msg

        # 上方消息文本（居中，占屏幕宽度80%）
        self.msg_text = MultiText(
            x=0.5, y=0.3, w=0.8, h=0.3, center=(0.5, 0.5),
            text=self.msg, color=(255, 255, 255), font_size=36
        )

        # 下方提示文本（居中）
        self.hint_text = MultiText(
            x=0.5, y=0.7, w=0.8, h=0.1, center=(0.5, 0.5),
            text="按下空格继续", color=(255, 255, 255), font_size=24
        )

    def event_parse(self, event):
        super().event_parse(event)  # 继承基础事件监听（如退出事件）

        # 检测空格键按下
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            self.running = False

    def draw(self):
        # 深色背景
        self.app.screen.fill((0, 0, 0))

        # 绘制文本
        self.msg_text.draw(self.app.screen)
        self.hint_text.draw(self.app.screen)


class WaitProgressBarScene(SceneBase):
    def __init__(self, app: App, msg: str, seconds: float):
        super().__init__(app)
        self.msg = msg
        self.duration = seconds * 1000  # 转换为毫秒
        self.start_time = 0
        self.progress = 0.0

        # 上方消息文本（居中）
        self.msg_text = MultiText(
            x=0.5, y=0.3, w=0.8, h=0.3, center=(0.5, 0.5),
            text=self.msg, color=(255, 255, 255), font_size=36
        )

        # 进度条（占屏幕宽度60%，高度2%）
        self.progress_bar = ProgressBar(
            x=0.5, y=0.7, w=0.4, h=0.05, center=(0.5, 0.5),
            border_color=(155, 155, 155), fill_color=(155, 155, 155)
        )

    def event_parse(self, event):
        super().event_parse(event)  # 继承基础事件监听（如退出事件）

        # 检测空格键按下
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            self.running = False

    def enter(self):
        self.start_time = pygame.time.get_ticks()
        super().enter()

    def update_progress(self):
        current_time = pygame.time.get_ticks()
        elapsed = current_time - self.start_time
        self.progress = min(1.0, elapsed / self.duration)

        # 更新进度条和百分比文本
        self.progress_bar.set_progress(self.progress)

        # 进度完成后自动退出
        if self.progress >= 1.0:
            self.running = False

    def draw(self):
        # 更新进度
        self.update_progress()

        # 深色背景
        self.app.screen.fill((0, 0, 0))

        # 绘制所有组件
        self.msg_text.draw(self.app.screen)
        self.progress_bar.draw(self.app.screen)


class Button(ComponentBase):
    def __init__(self, x=0., y=0., w=0., h=0., center=(0.5, 0.5),
                 color=(100, 100, 100), hover_color=(150, 150, 150),
                 text="", text_color=(255, 255, 255), font_size=24,
                 font_name="simsun"):
        super().__init__(x, y, w, h, center)
        self.color = color
        self.hover_color = hover_color
        self.text = text
        self.text_color = text_color
        self.font_size = font_size
        self.font_name = font_name
        self._font = None
        self._last_size = (0, 0)
        self._current_color = color
        self.rect = None

    def _update_font(self, screen_width, screen_height):
        """根据窗口大小动态调整字体"""
        scaled_size = max(12, int(self.font_size * min(screen_width, screen_height) / 800))
        if (screen_width, screen_height) != self._last_size or self._font is None:
            try:
                self._font = pygame.font.SysFont(self.font_name, scaled_size)
                self._last_size = (screen_width, screen_height)
            except:
                self._font = pygame.font.SysFont(None, scaled_size)

    def handle_event(self, event, screen_width, screen_height):
        abs_x, abs_y, abs_w, abs_h = self.get_coord(screen_width, screen_height)
        self.rect = pygame.Rect(abs_x, abs_y, abs_w, abs_h)

        if event.type == pygame.MOUSEMOTION:
            if self.rect.collidepoint(event.pos):
                self._current_color = self.hover_color
            else:
                self._current_color = self.color

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False

    def draw(self, screen):
        screen_width, screen_height = screen.get_size()
        self._update_font(screen_width, screen_height)
        abs_x, abs_y, abs_w, abs_h = self.get_coord(screen_width, screen_height)

        # 绘制按钮
        pygame.draw.rect(screen, self._current_color, (abs_x, abs_y, abs_w, abs_h))

        # 绘制按钮文字
        text_surf = self._font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=(abs_x + abs_w / 2, abs_y + abs_h / 2))
        screen.blit(text_surf, text_rect)


class InputBox(ComponentBase):
    def __init__(self, x=0., y=0., w=0., h=0., center=(0.5, 0.5),
                 color_inactive=(100, 100, 100), color_active=(150, 150, 150),
                 text_color=(255, 255, 255), font_size=24, font_name="simsun",
                 prompt=""):
        super().__init__(x, y, w, h, center)
        self.color_inactive = color_inactive
        self.color_active = color_active
        self.text_color = text_color
        self.font_size = font_size
        self.font_name = font_name
        self.prompt = prompt
        self.text = ""
        self.active = False
        self._font = None
        self._last_size = (0, 0)

    def _update_font(self, screen_width, screen_height):
        """根据窗口大小动态调整字体"""
        scaled_size = max(12, int(self.font_size * min(screen_width, screen_height) / 800))
        if (screen_width, screen_height) != self._last_size or self._font is None:
            try:
                self._font = pygame.font.SysFont(self.font_name, scaled_size)
                self._last_size = (screen_width, screen_height)
            except:
                self._font = pygame.font.SysFont(None, scaled_size)

    def handle_event(self, event, screen_width, screen_height):
        abs_x, abs_y, abs_w, abs_h = self.get_coord(screen_width, screen_height)
        rect = pygame.Rect(abs_x, abs_y, abs_w, abs_h)

        if event.type == pygame.MOUSEBUTTONDOWN:
            if rect.collidepoint(event.pos):
                self.active = True
                pygame.key.set_text_input_rect(rect)  # 设置输入区域
                pygame.key.start_text_input()  # 启用文本输入
            else:
                self.active = False
                pygame.key.stop_text_input()  # 停止文本输入

        # 处理功能键
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                pygame.key.stop_text_input()
                return True
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_v and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                # 处理Ctrl+V粘贴
                self.text += pygame.scrap.get(pygame.SCRAP_TEXT).decode('utf-8').strip('\x00')

        # 处理文本输入（这才是真正接收字符的地方）
        if event.type == pygame.TEXTINPUT and self.active:
            self.text += event.text

        return False

    def draw(self, screen):
        screen_width, screen_height = screen.get_size()
        self._update_font(screen_width, screen_height)
        abs_x, abs_y, abs_w, abs_h = self.get_coord(screen_width, screen_height)

        # 绘制输入框
        color = self.color_active if self.active else self.color_inactive
        pygame.draw.rect(screen, color, (abs_x, abs_y, abs_w, abs_h), 2)

        # 绘制提示文字
        if not self.text and not self.active:
            prompt_surf = self._font.render(self.prompt, True, (150, 150, 150))
            screen.blit(prompt_surf, (abs_x + 5, abs_y + (abs_h - prompt_surf.get_height()) // 2))

        # 绘制输入文字
        text_surf = self._font.render(self.text, True, self.text_color)
        screen.blit(text_surf, (abs_x + 5, abs_y + (abs_h - text_surf.get_height()) // 2))

class InputScene(SceneBase):
    def __init__(self, app: App, prompt="请输入:", button_text="确定"):
        super().__init__(app)
        self.input_text = ""

        # 创建输入框
        self.input_box = InputBox(
            x=0.5, y=0.4, w=0.6, h=0.1,
            center=(0.5, 0.5),
            prompt=prompt
        )

        # 创建确定按钮
        self.button = Button(
            x=0.5, y=0.6, w=0.2, h=0.1,
            center=(0.5, 0.5),
            text=button_text
        )

    def event_parse(self, event):
        super().event_parse(event)

        screen_width, screen_height = self.app.screen.get_size()

        # 处理输入框事件
        if self.input_box.handle_event(event, screen_width, screen_height):
            self.input_text = self.input_box.text
            self.running = False

        # 处理按钮事件
        if self.button.handle_event(event, screen_width, screen_height):
            self.input_text = self.input_box.text
            self.running = False

    def draw(self):
        self.app.screen.fill((30, 30, 30))
        self.input_box.draw(self.app.screen)
        self.button.draw(self.app.screen)

    def get_text(self):
        return self.input_text

class CrossHair(ComponentBase):
    def __init__(self, x=0.5, y=0.5, size=0.1, center=(0.5, 0.5), color=(255, 255, 255)):
        """
        :param size: 准星总尺寸，相对屏幕宽度，控制线的长度
        """
        super().__init__(x, y, 0, 0, center)
        self.size = size
        self.color = color

    def draw(self, screen):
        screen_width, screen_height = screen.get_size()
        cx, cy = self.proj(self.x, self.y, screen_width, screen_height)
        cross_len = int(screen_width * self.size)

        # 横线略长于竖线
        pygame.draw.line(screen, self.color, (cx - cross_len, cy), (cx + cross_len, cy), 4)
        pygame.draw.line(screen, self.color, (cx, cy - int(cross_len * 0.8)), (cx, cy + int(cross_len * 0.8)), 4)


class CrossScene(SceneBase):
    def __init__(self, app, prompt, second, sound=None):
        super().__init__(app)
        self.prompt = prompt
        self.second = second
        self.sound = sound
        self.start_time = None

        self.text = MultiText(
            x=0.5, y=0.3, w=0.8, h=0.3,
            center=(0.5, 0.5),
            text=self.prompt, color=(255, 255, 255), font_size=36
        )

        self.crosshair = CrossHair(x=0.5, y=0.5, size=0.09)

    def enter(self):
        if self.sound:
            try:
                self.sound.play()
            except Exception as e:
                print(f"播放音频失败: {e}")
        self.start_time = pygame.time.get_ticks()
        return super().enter()

    def draw(self):
        self.app.screen.fill((0, 0, 0))
        self.text.draw(self.app.screen)
        self.crosshair.draw(self.app.screen)

        elapsed = (pygame.time.get_ticks() - self.start_time) / 1000.0
        if elapsed >= self.second:
            self.running = False


class BlackScene(SceneBase):
    def __init__(self, app, second):
        super().__init__(app)
        self.second = second
        self.start_time = None

    def enter(self):
        self.start_time = pygame.time.get_ticks()
        return super().enter()

    def draw(self):
        self.app.screen.fill((0, 0, 0))
        elapsed = (pygame.time.get_ticks() - self.start_time) / 1000.0
        if elapsed >= self.second:
            self.running = False