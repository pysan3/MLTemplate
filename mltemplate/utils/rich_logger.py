# WARNING: Please do NOT edit this file unless you know what you are doing.

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Callable, Iterable, Optional

from rich.logging import RichHandler
from rich.text import Text, TextType

if TYPE_CHECKING:
    from rich.console import Console, ConsoleRenderable, RenderableType
    from rich.table import Table

FormatTimeCallable = Callable[[datetime.datetime], Text]


class MyLogRender:
    def __init__(
        self,
        show_time: bool = True,
        show_level: bool = False,
        show_path: bool = True,
        time_format: str | FormatTimeCallable = "[%x %X]",
        omit_repeated_times: bool = True,
        level_width: Optional[int] = 8,
    ) -> None:
        self.show_time = show_time
        self.show_level = show_level
        self.show_path = show_path
        self.time_format = time_format
        self.omit_repeated_times = omit_repeated_times
        self.level_width = level_width
        self._last_time = None

    def __call__(
        self,
        console: Console,
        renderables: Iterable[ConsoleRenderable],
        log_time: Optional[datetime.datetime] = None,
        time_format: Optional[str | FormatTimeCallable] = None,
        level: TextType = "",
        path: Optional[str] = None,
        line_no: Optional[int] = None,
        link_path: Optional[str] = None,
    ) -> Table:
        from rich.containers import Renderables
        from rich.table import Table

        output = Table.grid(padding=(0, 1))
        output.expand = True
        output.add_column(ratio=1, style="log.message", overflow="fold")
        if self.show_level:
            output.add_column(style="log.level", width=self.level_width)
        if self.show_path and path:
            output.add_column(style="log.path")
        if self.show_time:
            output.add_column(style="log.time")
        row: list[RenderableType] = [Renderables(renderables)]
        if self.show_level:
            row.append(level)
        if self.show_path and path:
            path_text = Text()
            path_text.append(f"link file://{link_path}" if link_path else "")
            if line_no:
                path_text.append(":")
                path_text.append(f"link file://{link_path}#{line_no}" if link_path else "")
        if self.show_time:
            log_time = log_time or console.get_datetime()
            time_format = time_format or self.time_format
            if callable(time_format):
                log_time_display = time_format(log_time)
            else:
                log_time_display = Text(log_time.strftime(time_format))
            if log_time_display == self._last_time and self.omit_repeated_times:
                row.append(Text(" " * len(log_time_display)))
            else:
                row.append(log_time_display)
                self._last_time = log_time_display
        output.add_row(*row)
        return output


class MyRichHandler(RichHandler):
    def __init__(
        self,
        *args,
        show_time: bool = True,
        omit_repeated_times: bool = True,
        show_level: bool = True,
        show_path: bool = True,
        enable_link_path: bool = True,
        log_time_format: str | FormatTimeCallable = "[%x %X]",
        **kwargs,
    ) -> None:
        kwargs["show_time"] = show_time
        kwargs["omit_repeated_times"] = omit_repeated_times
        kwargs["show_level"] = show_level
        kwargs["show_path"] = show_path
        kwargs["enable_link_path"] = enable_link_path
        kwargs["log_time_format"] = log_time_format
        super().__init__(*args, **kwargs)
        self._log_render = MyLogRender(
            show_time=show_time,
            show_level=show_level,
            show_path=show_path,
            time_format=log_time_format,
            omit_repeated_times=omit_repeated_times,
            level_width=None,
        )