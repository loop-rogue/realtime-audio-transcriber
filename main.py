import sounddevice as sd
import numpy as np
from whispercpp import Whisper
from whispercpp import api
from queue import Queue
import threading
import time
from numpy.typing import NDArray
import traceback
from typing import Tuple, Optional
import curses
import ollama
import copy
import multiprocessing
import re
from wcwidth import wcswidth
import datetime
import soundfile as sf
from pathlib import Path


import logging
from logging.handlers import RotatingFileHandler
import os
import sys


param = (
    api.Params.from_enum(api.SAMPLING_GREEDY)
    .with_print_progress(False)
    .with_max_segment_length(1)
    .with_token_timestamps(True)
)


# 音频配置
SAMPLE_RATE = 16000
CHUNK_DURATION = 2  # Duration of each audio chunk (seconds)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
AUDIO_FORMAT = "float32"
DEVICE = "BlackHole 2ch"

audio_queue = Queue(maxsize=20)
# global queue: used to decouple the file writing task from the processing logic
file_write_queue = Queue(maxsize=10)

# recorded autio data
recorded_audio_data = []

logger = logging.getLogger(__name__)


def setup_session_logging(output_folder: Path):
    """Setup session-specific logging"""
    log_format = "%(asctime)s - %(levelname)s - %(processName)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # Create session log file
    session_log_file = output_folder / "session.log"
    session_handler = logging.FileHandler(session_log_file, encoding="utf-8")
    session_handler.setFormatter(formatter)

    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(session_handler)

    logger.info(f"Session logging started - log file: {session_log_file}")
    return session_log_file


def validate_startup_requirements():
    """Validate requirements before starting TUI"""
    errors = []
    warnings = []

    # Check if Ollama is running (optional - show warning only)
    try:
        ollama.list()
        logger.info("Ollama connection successful - translation enabled")
    except Exception as e:
        warning_msg = f"⚠️  Ollama not accessible: {e}"
        warnings.append(warning_msg)
        warnings.append(
            "   Translation will be disabled. To enable: 'ollama serve' or install from https://ollama.ai"
        )
        logger.warning(f"Ollama not available: {e}")

    # Check audio device (critical)
    try:
        devices = sd.query_devices()
        device_names = [d["name"] for d in devices]
        if DEVICE not in device_names:
            errors.append(f"Audio device '{DEVICE}' not found")
            errors.append(f"Available devices: {device_names}")
            errors.append(
                "Install BlackHole 2ch: https://github.com/ExistentialAudio/BlackHole"
            )
    except Exception as e:
        errors.append(f"Audio device check failed: {e}")
        errors.append(
            "Install BlackHole 2ch: https://github.com/ExistentialAudio/BlackHole"
        )

    return errors, warnings


def suppress_output():
    """Redirect stdout and stderr to prevent interference with TUI"""
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")


def restore_output():
    """Restore stdout and stderr"""
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


class CursesUI:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.lock = threading.Lock()
        self.lines = []
        self.offset_y = 0

        curses.start_color()
        curses.use_default_colors()
        # init color pair 1: foreground yellow, background default
        curses.init_pair(1, curses.COLOR_YELLOW, -1)

    def update_line(self, line_num: int, text: str):
        try:
            with self.lock:
                self._update_text_line(line_num, text)

                max_y, _ = self.stdscr.getmaxyx()

                index = line_num - self.offset_y
                if index < max_y:
                    self.stdscr.move(index, 0)
                    self.stdscr.clrtoeol()
                    self._print_line_with_highlighted_text(index, text)
                    self.stdscr.refresh()
                else:
                    # TODO: need update better here
                    # assume index == max_y
                    self.offset_y += 1

                    for index, line in enumerate(
                        self.lines[self.offset_y : self.offset_y + max_y]
                    ):
                        self.stdscr.move(index, 0)
                        self.stdscr.clrtoeol()
                        self._print_line_with_highlighted_text(index, line)
                    self.stdscr.refresh()
        except:
            logger.error("Error update_line", exc_info=True)

    def _update_text_line(self, line_num: int, text: str):
        while line_num >= len(self.lines):
            self.lines.append("")
        # line_num < len(self.line)
        self.lines[line_num] = text

    def _print_line_with_highlighted_text(self, line: int, text: str, color_pair=1):
        segments = re.split(r"(\*\*.*?\*\*)", text)
        current_col = 0
        for seg in segments:
            if seg.startswith("**") and seg.endswith("**"):
                # 去掉 ** 标记
                colored_text = seg[2:-2]
                self.stdscr.addstr(
                    line, current_col, colored_text, curses.color_pair(color_pair)
                )
                # use wcswidth to accurately calculate the display width of each string, because Chinese characters usually display as 2 character width
                current_col += wcswidth(colored_text)
            else:
                self.stdscr.addstr(line, current_col, seg)
                # use wcswidth to accurately calculate the display width of each string, because Chinese characters usually display as 2 character width
                current_col += wcswidth(seg)

    def clear(self):
        with self.lock:
            self.stdscr.clear()


class TimedTranscription:
    def __init__(self, start_ts, end_ts, text):
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.text = text

    def __str__(self):
        return f"[{self.start_ts} --> {self.end_ts}] {self.text}"

    def __repr__(self):
        # return f"TimedTranscription(start_ts={self.start_ts}, end_ts={self.end_ts}, text={repr(self.text)})"
        return f"[{self.start_ts} --> {self.end_ts}] {self.text}"

    def is_end_punctuation(self):
        return (
            self.text.strip() == "."
            or self.text.strip() == ","
            or self.text.strip() == "]"
        )

    def is_at_ts(self, ts):
        return self.start_ts <= ts and ts <= self.end_ts

    def is_empty(self):
        return self.text.strip() == ""


def to_time(n):
    return time.strftime("%H:%M:%S", time.gmtime(int(n / 100)))


def list_rfind(lst, condition):
    return next((i for i in reversed(range(len(lst))) if condition(lst[i])), -1)


def transcribe_with_time(
    self, data: NDArray[np.float32], num_proc: int = 1, strict: bool = False
):
    if strict:
        assert (
            self.context.is_initialized
        ), "strict=True and context is not initialized. Make sure to call 'context.init_state()' before."
    else:
        if not self.context.is_initialized and not self._context_initialized:
            self.context.init_state()
            self._context_initialized = True

    self.context.full_parallel(self.params, data, num_proc)

    def safe_get_segment_text(i):
        try:
            text = self.context.full_get_segment_text(i)
            # Handle case where text might be bytes
            if isinstance(text, bytes):
                return text.decode("utf-8", errors="replace")
            return text if text is not None else ""
        except (UnicodeDecodeError, AttributeError):
            logger.warning(f"Failed to decode segment text at index {i}")
            return ""

    return [
        TimedTranscription(
            start_ts=self.context.full_get_segment_start(i),
            end_ts=self.context.full_get_segment_end(i),
            text=safe_get_segment_text(i),
        )
        for i in range(self.context.full_n_segments())
    ]


# time stamp is the ts in the current buffer
def ts_to_buffer_index(ts):
    # the number of data in one unit of ts. (ts is the unit of 1/100s)
    # SAMPLE_RATE is the number of data in one second. (each f32 is one data)
    cout_per_unit = SAMPLE_RATE / 100
    return int(ts * cout_per_unit)


Whisper.transcribe_with_time = transcribe_with_time


# both trimed results
# return None if current_timed_results is empty
def find_next_confirmed_end_ts(
    last_timed_results: list[TimedTranscription],
    current_timed_results: list[TimedTranscription],
) -> Tuple[Optional[int], Optional[int]]:
    if len(last_timed_results) == 0:
        return (None, None)
    first_word_in_current_index = next(
        (i for i, x in enumerate(current_timed_results) if not x.is_empty()), -1
    )
    if first_word_in_current_index == -1:
        # print("first_word_in_current_index is -1")
        return (None, None)
    first_word = current_timed_results[first_word_in_current_index].text.strip().lower()

    # print(f"find the first word: {first_word}")
    first_word_in_last_index = next(
        (
            i
            for i, x in enumerate(last_timed_results)
            if x.text.strip().lower() == first_word
        ),
        -1,
    )
    if first_word_in_last_index == -1:
        # print(last_timed_results)
        # print("first_word_in_last_index is -1")
        return (None, None)

    # TODO: check the first_word_in_current_index and first_word_in_last_index 's start ts and end ts

    # get the longest common prefix starting from first_word_in_current_index

    last_filtered = list(
        filter(
            lambda x: not x.is_empty(), last_timed_results[first_word_in_last_index:]
        )
    )
    current_filtered = list(
        filter(
            lambda x: not x.is_empty(),
            current_timed_results[first_word_in_current_index:],
        )
    )

    next_confirmed = None
    end_punctuation = None
    for l, c in zip(last_filtered, current_filtered):
        if l.text.strip().lower() == c.text.strip().lower():
            next_confirmed = c

            # if we have . or , or ] in the confirmed text, we will do the trimming the audio buffer
            if c.is_end_punctuation():
                end_punctuation = c
        else:
            break
    if next_confirmed != None:
        end_punctuation_end_ts = None
        if end_punctuation != None:
            end_punctuation_end_ts = end_punctuation.end_ts
        return next_confirmed.end_ts, end_punctuation_end_ts
    else:
        # print("next_confirmed is None")
        return (None, None)


def audio_callback(indata, frames, time, status):
    """音频输入回调函数"""
    if status:
        print(f"Audio stream status: {status}")
    audio_queue.put(indata[:, 0].copy())  # 单声道

    recorded_audio_data.append(indata.copy())  # store data


def reset_timed_output(timed_output):
    start_ts = timed_output[0].start_ts
    new_timed_output = timed_output[:]
    for i in range(len(new_timed_output)):
        new_timed_output[i].start_ts -= start_ts
        new_timed_output[i].end_ts -= start_ts
    return new_timed_output


def timed_results_from(timed_results, ts):
    index = next((i for i, x in enumerate(timed_results) if x.is_at_ts(ts)), -1)
    if index == -1:
        return []
    else:
        return timed_results[index:]


def cut_timed_results_to_ts(timed_results, ts):
    index = next((i for i, x in enumerate(timed_results) if x.is_at_ts(ts)), -1)
    if index == -1:
        return []
    else:
        return timed_results[: index + 1]


def process_audio(curse_ui: CursesUI, conn, model: Whisper):
    """处理音频的线程函数"""
    logger.info("Starting audio processing thread")

    curse_ui.clear()

    buffer = np.array([], dtype=AUDIO_FORMAT)

    last_confirmed_end_ts = -1
    # the buffer's last transcribe result which contains the timestamp
    last_buffered_time_res = []

    current_console_line = 0

    while True:
        try:
            # get audio data
            data = audio_queue.get(timeout=1)
            buffer = np.concatenate([buffer, data])

            buffer_len_in_seconds = len(buffer) / SAMPLE_RATE

            logger.debug(
                f"Audio buffer updated. Current size (seconds): {len(buffer)/SAMPLE_RATE:.2f}s"
            )

            # process the audio chunk if it's long enough
            if len(buffer) >= CHUNK_SIZE:
                # extract the current chunk and keep 1 second context
                # segment, buffer = buffer[:CHUNK_SIZE], buffer[-SAMPLE_RATE:]

                segment = buffer

                logger.debug(
                    f"Starting transcription for {len(segment)/SAMPLE_RATE:.2f}s audio"
                )

                # transcribe the segment
                transcribe_start = time.time()
                time_result = model.transcribe_with_time(segment)
                logger.info(
                    f"Transcription completed. Segments: {len(time_result)} | Time: {(time.time()-transcribe_start)*1000:.2f}ms"
                )

                if last_confirmed_end_ts < 0:
                    last_confirmed_end_ts = 0
                    last_buffered_time_res = time_result
                    continue

                # starting from last confirmed ts of both current transcription and last transcription
                # to find the next confirmed ts (using longest common prefix for the confirm)
                # through the iteration it will also get the end_punctuation_end_ts which is the 标点符号(, or .) to split the transcription
                last = timed_results_from(
                    last_buffered_time_res, max(last_confirmed_end_ts - 1 * 100, 0)
                )
                current = timed_results_from(time_result, last_confirmed_end_ts)
                next_confirmed_ts, end_punctuation_end_ts = find_next_confirmed_end_ts(
                    last, current
                )

                if next_confirmed_ts == None:
                    last_buffered_time_res = time_result
                    # print("next_confirmed_ts is None")
                    continue

                # confirmed_s = cut_timed_results_to_ts(time_result, next_confirmed_ts)
                # print("======= confirmed ==========")
                # print("".join([x.text for x in confirmed_s]))

                text = "".join([x.text for x in time_result])
                curse_ui.update_line(current_console_line * 3, text)
                logger.debug(f"UI updated for line {current_console_line}")

                # if the buffer is longer than a certain time length, we will cut the buffer at the confirmed ts
                if buffer_len_in_seconds > 20:
                    end_punctuation_end_ts = next_confirmed_ts
                    # if the confirmed text is still very short, just cut the whole text
                    # (ts is the unit of 1/100s)
                    if next_confirmed_ts / 100 < 10:
                        next_confirmed_ts = time_result[-1].end_ts
                        end_punctuation_end_ts = next_confirmed_ts

                # handle the cut
                if end_punctuation_end_ts == None:
                    # just update the confirmed ts and use the current transcripe result
                    last_confirmed_end_ts = next_confirmed_ts
                    last_buffered_time_res = time_result
                else:
                    # cut the buffer
                    partitionan_index = ts_to_buffer_index(end_punctuation_end_ts)
                    buffer = buffer[partitionan_index:]

                    # cut the text
                    confirmed = cut_timed_results_to_ts(
                        time_result, end_punctuation_end_ts
                    )
                    text = "".join([x.text for x in confirmed])
                    curse_ui.update_line(current_console_line * 3, text)

                    conn.send((current_console_line, copy.copy(text)))

                    # send to file_write_queue to write to file
                    file_write_queue.put((current_console_line, text))

                    # update the new confirmed_ts
                    last_confirmed_end_ts = next_confirmed_ts - end_punctuation_end_ts
                    last_buffered_time_res = timed_results_from(
                        time_result, end_punctuation_end_ts
                    )

                    current_console_line += 1
                    text = "".join([x.text for x in last_buffered_time_res])
                    curse_ui.update_line(current_console_line * 3, text)

                    # since we reset the buffer, we need to update the timestamp of the transcripe result
                    last_buffered_time_res = reset_timed_output(last_buffered_time_res)

        except Exception as e:
            logger.critical("Fatal error in audio processing thread", exc_info=True)
            traceback.print_exc()


def process_output_from_ai_translate(curse_ui: CursesUI, conn):
    logger.info("Starting translation output thread")
    while True:
        try:
            index, translate = conn.recv()
            lines = translate.split("\n")

            def _update_for_line(content_lines):
                curse_ui.update_line(3 * index, content_lines[0].strip())
                curse_ui.update_line(3 * index + 1, content_lines[1].strip())

            logger.debug(f"Received translation for line {index}, lines: {lines})")
            if len(lines) == 2:
                _update_for_line(lines)
            else:
                logger.warning(
                    f"Received translation is not 2 lines, get translation content: \n{translate}"
                )
                # Some fallback
                lines = list(filter(lambda line: line.strip() != "", lines))
                if len(lines) == 2:
                    logging.info("Fall back to have empty lines cases")
                    _update_for_line(lines)
                    continue
                content = "\n".join(lines)

                def _extract_first_code_block(text):
                    match = re.search(r"```([\s\S]*?)```", text)
                    return match.group(1).strip() if match else None

                block = _extract_first_code_block(content).strip()
                new_lines = block.split("\n")
                if len(new_lines) == 2:
                    logging.info("Fall back to have code block in results")
                    _update_for_line(new_lines)
                    continue
                logger.error(
                    f"No fall back get the translate/highlight lines for: \n{translate}"
                )
        except:
            logger.critical(
                "Fatal error in process_output_from_ai_translate thread", exc_info=True
            )


def process_transcription(conn):
    # Suppress stdout/stderr in this multiprocess too
    suppress_output()

    logger.info("Starting transcription processing process")
    while True:
        try:
            index, transcription = conn.recv()

            logger.info(
                f"Received transcription for line {index}: {transcription[:50]}..."
            )

            start_time = time.time()

            response = ollama.generate(
                # model='qwen2.5:14b',
                model="gemma3:27b",
                prompt=f"""
你是一个高效的信息提取助手。我将提供给你一个实时转录的英文句子，该句子可能语速快、信息密集且带有口语化表达。请按以下要求处理这个句子：

1. **原句输出**：在第一行输出原始英文句子，但请你识别出句子中最关键的信息或部分，并将这些关键部分使用 `**` 标记加粗，便于快速抓住重点, 比如主谓宾。

2. **翻译输出**：在第二行输出该英文句子的中文翻译。翻译时，请确保对应的重点部分也同样用 `**` 标记加粗，保持与原句中加粗部分的一一对应，帮助快速理解重点内容。

例如：
输入句子：
```
The quick brown fox jumps over the lazy dog.
```
可能的输出：
```
The **quick brown** fox jumps over the **lazy dog**.
快速的**棕色狐狸**跃过了**懒狗**.
```
请确保每个需要加粗的部分都严格以 `**` 开始和结束，不要在加粗标记内部添加多余的空格或符号。

如果给的句子没有实际内容，就直接给出原来结果，然后翻译也就给原文.
如果不是个完整句子，就加粗你觉得重要的部分，能翻译就翻译，翻译不了第二行给原文

这个是需要处理的句子
```
{transcription}
```
""",
            )
            logger.info(
                f"Processed translation for line {index} in {(time.time()-start_time)*1000:.2f}ms"
            )

            response = response["response"]
            response = response.strip("` \n")
            conn.send((index, response))

        except Exception as e:
            logger.error(f"Translation failed for line {index}: {e}")


def process_file_writer(file_write_queue: Queue, transcribe_output_file: Path):
    """
    File writer thread: Creates a new file on each program execution (stored in the 'transcripts' folder
    in the current directory, with filename prefixed by date and time). Subsequently, transcription and
    translation results will be appended to this file.
    """
    folder = "transcripts"
    if not os.path.exists(folder):
        os.makedirs(folder)
    logger.info(f"Transcript file: {transcribe_output_file}")
    try:
        with open(transcribe_output_file, "w", encoding="utf-8") as f:
            while True:
                # 阻塞等待写入任务
                index, original = file_write_queue.get()
                f.write(f"Line {index}:\n")
                f.write(original + "\n")
                f.write("-" * 40 + "\n")
                f.flush()
    except Exception as e:
        logger.critical("Error in file writer thread", exc_info=True)


def main(stdscr, audio_file: Path, transcribe_file: Path):

    logger.info("Application starting")

    # suppress all stdout/stderr output when TUI starts
    suppress_output()

    # init Whisper model after output suppression, avoid model loading info to interrupt TUI
    logger.info("Loading Whisper model...")
    model = Whisper.from_params("tiny.en", params=param)
    logger.info("Whisper model loaded successfully")

    # 配置音频输入流
    stream = sd.InputStream(
        device=DEVICE,
        samplerate=SAMPLE_RATE,
        channels=2,
        dtype=AUDIO_FORMAT,
        blocksize=SAMPLE_RATE // 2,  # 0.5 second chunk size, each block is 0.5 second
        # blocksize=SAMPLE_RATE,  # 1 second chunk size, each block is 1 second
        callback=audio_callback,
    )

    ui = CursesUI(stdscr)

    parent_conn, child_conn = multiprocessing.Pipe(duplex=True)

    # start the processing thread
    processing_thread = threading.Thread(
        target=process_audio, args=(ui, parent_conn, model), daemon=True
    )
    processing_thread.start()

    processing_translate_thread = threading.Thread(
        target=process_output_from_ai_translate, args=(ui, parent_conn), daemon=True
    )
    processing_translate_thread.start()

    # start the file writing thread
    file_writer_thread = threading.Thread(
        target=process_file_writer,
        args=(file_write_queue, transcribe_file),
        daemon=True,
    )
    file_writer_thread.start()

    new_proc = multiprocessing.Process(target=process_transcription, args=(child_conn,))
    new_proc.start()

    try:
        # start the audio stream
        logger.info("Starting audio stream")

        with stream:
            # Update UI instead of print to stdout
            ui.update_line(
                0,
                "=== Starting Real-time Speech Recognition (LocalAgreement-2 Optimized) ===   Press Ctrl+C to stop...",
            )
            while True:
                threading.Event().wait(1)

    except KeyboardInterrupt:
        logger.info("User requested stop")

    finally:
        # when TUI ends, restore stdout/stderr
        restore_output()

        # Convert recorded data to a NumPy array and save
        if recorded_audio_data:
            audio_array = np.concatenate(recorded_audio_data, axis=0)
            sf.write(audio_file, audio_array, SAMPLE_RATE)
            logger.info(f"Audio saved to {audio_file}")


def startup_main():
    """Main function with startup validation"""
    # Validate requirements before starting TUI
    errors, warnings = validate_startup_requirements()

    # Show warnings (but don't stop)
    if warnings:
        print("⚠️  Warnings:")
        for warning in warnings:
            print(f"  {warning}")
        print()

    # Stop only on critical errors
    if errors:
        print("❌ Critical errors - cannot start:")
        for error in errors:
            print(f"  • {error}")
        print("\nPlease fix these issues and try again.")
        sys.exit(1)

    print("✅ Startup validation complete")
    if warnings:
        print("🎙️  Starting transcription-only mode...")
    else:
        print("🚀 Starting transcription + translation mode...")

    # Create output folder structure
    output_folder = (
        Path(__file__).parent
        / "output"
        / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    audio_file = output_folder / "audio.wav"
    transcribe_file = output_folder / "transcribe.txt"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Setup session logging
    session_log_file = setup_session_logging(output_folder)

    # Start TUI
    curses.wrapper(main, audio_file, transcribe_file)

    # After TUI exits, show file summary
    print("\n" + "=" * 60)
    print("📁 Session Complete - Files Saved:")
    print("=" * 60)

    if audio_file.exists():
        print(f"🎵 Audio: {audio_file}")
    else:
        print("🎵 Audio: No audio recorded")

    if transcribe_file.exists():
        print(f"📝 Transcript: {transcribe_file}")
    else:
        print("📝 Transcript: No transcript saved")

    # Show session log file
    if session_log_file and session_log_file.exists():
        print(f"📋 Session Log: {session_log_file}")

    print(f"📂 Session folder: {output_folder}")
    print("=" * 60)


if __name__ == "__main__":
    startup_main()
