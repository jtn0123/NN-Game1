"""
Web Module
==========

Flask-based web dashboard for remote monitoring of training.

Components:
    server.py    - Flask + SocketIO server
    templates/   - HTML templates
    static/      - CSS and JavaScript
"""

from .server import WebDashboard, MetricsPublisher

__all__ = ['WebDashboard', 'MetricsPublisher']

