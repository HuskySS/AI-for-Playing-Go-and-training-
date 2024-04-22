from pgutils.pgcontrols.ctbase import CtBase
from pgutils.pgtools.toolbase import ToolBase
import pygame
from typing import List, Union
class Manager:
    def __init__(self):
        self.controls = []
        self.tools = []

    def control_register(self, controls: Union[List[CtBase], CtBase]):

        if isinstance(controls, CtBase):
            self.controls.append(controls)
        else:
            for control in controls:
                self.controls.append(control)

    def tool_register(self, tools: Union[List[ToolBase], ToolBase]):

        if isinstance(tools, ToolBase):
            self.tools.append(tools)
        else:
            for tool in tools:
                self.tools.append(tool)

    def control_update(self, event: pygame.event):

        for control in self.controls:
            if control.active:
                control.update(event)

    def tool_update(self):
        for tool in self.tools:
            if tool.active:
                tool.update()
                tool.disable()
