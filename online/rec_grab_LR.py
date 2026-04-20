import pygame.mixer
import os
from record.eeg_recorder_utils import EEGRecorder
from record.pygame_utils import *
from myutils import ZipStorage
import time
import random
class BiColorCrossHair(ComponentBase):
    def __init__(self, x=0.5, y=0.5, size=0.1, center=(0.5, 0.5),
                 color1=(0, 0, 0), color2=(255, 255, 255), alpha=0.0):
        """
        :param size: 准星总尺寸，相对屏幕宽度，控制线的长度
        :param color1: 第一种颜色 (默认黑色)
        :param color2: 第二种颜色 (默认白色)
        :param alpha: 混合程度 (0-1), 0表示完全color1，1表示完全color2
        """
        super().__init__(x, y, 0, 0, center)
        self.size = size
        self.color1 = color1
        self.color2 = color2
        self.alpha = max(0.0, min(1.0, alpha))  # 确保在0-1范围内

    def set_alpha(self, alpha):
        """设置混合程度 (0-1)"""
        self.alpha = max(0.0, min(1.0, alpha))

    def draw(self, screen):
        # 计算混合后的颜色
        r = int(self.color1[0] * (1 - self.alpha) + self.color2[0] * self.alpha)
        g = int(self.color1[1] * (1 - self.alpha) + self.color2[1] * self.alpha)
        b = int(self.color1[2] * (1 - self.alpha) + self.color2[2] * self.alpha)
        color = (r, g, b)

        screen_width, screen_height = screen.get_size()
        cx, cy = self.proj(self.x, self.y, screen_width, screen_height)
        cross_len = int(screen_width * self.size)

        # 绘制准星
        pygame.draw.line(screen, color, (cx - cross_len, cy), (cx + cross_len, cy), 4)
        pygame.draw.line(screen, color, (cx, cy - int(cross_len * 0.8)), (cx, cy + int(cross_len * 0.8)), 4)


class FlashCrossScene(SceneBase):
    def __init__(self, app, prompt, alpha_list, time_list, sound=None):
        """
        :param app: App实例
        :param prompt: 提示文本
        :param alpha_list: alpha值列表，如[0, 1, 0]
        :param time_list: 对应的时间点列表，如[0, 2, 4]
        :param sound: 可选的声音文件
        """
        super().__init__(app)
        self.prompt = prompt
        self.alpha_list = alpha_list
        self.time_list = time_list
        self.sound = sound
        self.start_time = None

        # 上方提示文本
        self.text = MultiText(
            x=0.5, y=0.48, w=0.8, h=0.3,
            center=(0.5, 0.5),
            text=self.prompt, color=(166, 166, 166), font_size=36
        )

        # 下方双色准星
        self.bicolor_crosshair = BiColorCrossHair(
            x=0.5, y=0.5, size=0.15,
            color1=(40, 40, 40), color2=(188, 188, 188), alpha=0.0
        )

    def enter(self):
        if self.sound:
            try:
                self.sound.play()
            except Exception as e:
                print(f"播放音频失败: {e}")
        self.start_time = pygame.time.get_ticks()
        return super().enter()

    def get_current_alpha(self):
        """根据当前时间计算alpha值"""
        current_time = (pygame.time.get_ticks() - self.start_time) / 1000.0

        # 如果超过最后一个时间点，使用最后一个alpha值
        if current_time >= self.time_list[-1]:
            return self.alpha_list[-1]

        # 找到当前时间所在的区间
        for i in range(len(self.time_list) - 1):
            if self.time_list[i] <= current_time <= self.time_list[i + 1]:
                # 计算线性插值
                t = (current_time - self.time_list[i]) / (self.time_list[i + 1] - self.time_list[i])
                return self.alpha_list[i] * (1 - t) + self.alpha_list[i + 1] * t

        return 0.0  # 默认值

    def draw(self):
        self.app.screen.fill((0, 0, 0))

        # 更新准星颜色
        current_alpha = self.get_current_alpha()
        self.bicolor_crosshair.set_alpha(current_alpha)

        # 绘制组件
        self.bicolor_crosshair.draw(self.app.screen)
        self.text.draw(self.app.screen)

        # 检查是否结束
        elapsed = (pygame.time.get_ticks() - self.start_time) / 1000.0
        if elapsed >= self.time_list[-1]:
            self.running = False

    def event_parse(self, event):
        super().event_parse(event)  # 继承基础事件监听（如退出事件）

        # 检测CTRL按下
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            self.running = False

if __name__ == '__main__':
    Subject_name = 'ZJY'

    all_tasks = []
    pygame.mixer.init()
    sound_left = pygame.mixer.Sound("voice_left.mp3")
    sound_right = pygame.mixer.Sound("voice_right.mp3")
    task_groups = {
        'mi': ['miL'] * 5 + ['miR'] * 5,
        # 'me': ['meL'] * 5 + ['meR'] * 5,
        'teethmi': ['teethmiL'] * 5 + ['teethmiR'] * 5,
        # 'teethme': ['teethmeL'] * 5 + ['teethmeR'] * 5,
        'blinkmi': ['blinkmiL'] * 5 + ['blinkmiR'] * 5,
        # 'blinkme': ['blinkmeL'] * 5 + ['blinkmeR'] * 5,
        'sweatmi': ['sweatmiL'] * 5 + ['sweatmiR'] * 5,
        # 'sweatme': ['sweatmeL'] * 5 + ['sweatmeR'] * 5,
        'shakemi': ['shakemiL'] * 5 + ['shakemiR'] * 5,
        # 'shakeme': ['shakemeL'] * 5 + ['shakemeR'] * 5,
        'rest': [],
        'presweat': [],
        'preshake': [],
        'end': [],
    }
    select_groups = (['mi', 'rest'] * 6 + ['teethmi', 'rest', 'blinkmi', 'preshake', 'shakemi']) * 2 + \
                    ['presweat'] + \
                    ['sweatmi'] * 2 + \
                    ['end']

    random.seed(0)
    for group in select_groups:
        cur_task = task_groups[group].copy()
        random.shuffle(cur_task)
        all_tasks += [f'tip_{group}']
        all_tasks += cur_task

    record_eeg = True
    time_name = time.time()
    save_directory = "C:/Users/ChenX-Lab/Desktop/Chen-X Lab MI Dataset/Results"
    os.makedirs(save_directory, exist_ok=True)  # 如果目录不存在则创建
    filename = os.path.join(save_directory, f"Record_MI_{time_name}_{Subject_name}.txt")  # 完整的文件路径
    app = App(1024, 768, "Grab LR")
    rec = EEGRecorder(32, 1000)
    cp = ZipStorage(os.path.join(save_directory, Subject_name), lock=False)
    if record_eeg:
        rec.start()
        cp.init()

    def rr(typ, dur):
        tm = time.time()
        if record_eeg:
            eeg = rec.get_record(dur)
            cp.append((tm-dur, tm), (typ, eeg))
        with open(filename, "a") as f:
            print(time.time(), typ, file=f, sep="\t")

    WaitScene(app, "Grab LR").enter()
    if record_eeg:
        WaitProgressBarScene(app, "正在检测脑电信号是否正常",5).enter()
        eeg = rec.get_record(5)
        print(eeg.var(axis=-1))
        WaitScene(app, "请检查信号是否正常").enter()

    total_ind = len(all_tasks)
    for task_ind, task in enumerate(all_tasks):
        if task=="tip_mi":
            WaitProgressBarScene(app, f"{task_ind}/{total_ind} 下面即将进行：运动想象",5).enter()
        if task=="tip_me":
            WaitProgressBarScene(app, f"{task_ind}/{total_ind}下面即将进行：运动执行",5).enter()
        if task=="tip_teethmi":
            WaitProgressBarScene(app, f"{task_ind}/{total_ind}下面即将进行：咬牙运动想象，请持续咬牙动作",5).enter()
        if task=="tip_teethme":
            WaitProgressBarScene(app, f"{task_ind}/{total_ind}下面即将进行：咬牙运动执行，请持续咬牙动作",5).enter()
        if task=="tip_blinkmi":
            WaitProgressBarScene(app, f"{task_ind}/{total_ind}下面即将进行：眨眼运动想象，请持续眨眼动作",5).enter()
        if task=="tip_blinkme":
            WaitProgressBarScene(app, f"{task_ind}/{total_ind}下面即将进行：眨眼运动执行，请持续眨眼动作",5).enter()
        if task=="tip_preshake":
            WaitScene(app, f"请主试同学负责模拟振动环境！").enter()
        if task=="tip_shakemi":
            WaitProgressBarScene(app, f"{task_ind}/{total_ind}下面即将进行：震动环境运动想象",5).enter()
        if task=="tip_shakeme":
            WaitProgressBarScene(app, f"{task_ind}/{total_ind}下面即将进行：震动环境运动执行",5).enter()
        if task=="tip_presweat":
            WaitScene(app, f"请在主试同学的辅助下进行运动活动，模拟出汗条件!").enter()
        if task=="tip_sweatmi":
            WaitProgressBarScene(app, f"{task_ind}/{total_ind}下面即将进行：出汗运动想象", 5).enter()
        if task=="tip_sweatme":
            WaitProgressBarScene(app, f"{task_ind}/{total_ind}下面即将进行：出汗运动执行", 5).enter()
        if task=='tip_rest':
            WaitScene(app, f"休息一下！").enter()
        if task.endswith("L") or task.endswith("R"):
            endtyp = task[-1]
            FlashCrossScene(app, "左手" if endtyp=='L' else "右手",
                            [0,0],[0,2],
                            sound_left if endtyp=='L' else sound_right).enter()
            FlashCrossScene(app, "", [0,1,0],[0,2,4]).enter()
            rr(task, 6)
        if task=='tip_end':
            WaitScene(app, "试验结束").enter()

    app.quit()

