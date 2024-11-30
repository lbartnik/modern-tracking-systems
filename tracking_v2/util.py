import plotly.graph_objects as go

class SubFigure:
    def __init__(self, fig: go.Figure, row: int, col: int):
        self.fig = fig
        self.row = row
        self.col = col
    
    def add_trace(self, *args, **kwargs):
        kwargs['row'] = self.row
        kwargs['col'] = self.col
        self.fig.add_trace(*args, **kwargs)
    
    def add_hline(self, *args, **kwargs):
        kwargs['row'] = self.row
        kwargs['col'] = self.col
        self.fig.add_hline(*args, **kwargs)
    
    def add_vline(self, *args, **kwargs):
        kwargs['row'] = self.row
        kwargs['col'] = self.col
        self.fig.add_vline(*args, **kwargs)

    def add_annotation(self, *args, **kwargs):
        kwargs['row'] = self.row
        kwargs['col'] = self.col
        self.fig.add_annotation(*args, **kwargs)
