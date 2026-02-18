#!/usr/bin/env python3
"""
Automated Report Generator for ML Stock Forecasting
Creates comprehensive PDF/HTML reports with charts, tables, and analysis
"""
import argparse
import os
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import base64
import json

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_pdf_availability():
    """Check if PDF generation is available"""
    try:
        from reportlab.lib import colors
        return True
    except ImportError:
        return False

def load_metrics_data(ticker: str) -> dict:
    """Load all metrics and results data"""
    data = {
        'ticker': ticker,
        'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': {},
        'next_day_prediction': {},
        'plots': {}
    }

    # Load ML metrics
    metrics_path = Path(f'src/reports/{ticker.lower()}_ml_metrics_summary.txt')
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            content = f.read()
            # Parse metrics from text file
            lines = content.split('\n')
            current_model = None
            for line in lines:
                line = line.strip()
                if line == 'Baseline:':
                    current_model = 'Baseline'
                    data['metrics']['Baseline'] = {}
                elif line.endswith('_Returns:'):
                    current_model = line.replace('_Returns:', '')
                    data['metrics'][current_model] = {}
                elif line and ':' in line and current_model:
                    key, value = line.split(':', 1)
                    try:
                        data['metrics'][current_model][key.strip()] = float(value.strip())
                    except ValueError:
                        data['metrics'][current_model][key.strip()] = value.strip()
    else:
        logging.warning(f"Metrics file not found: {metrics_path}")

    # Load next day prediction
    prediction_path = Path(f'src/reports/{ticker.lower()}_next_day_prediction.txt')
    if prediction_path.exists():
        with open(prediction_path, 'r') as f:
            content = f.read()
            # Extract key information
            lines = content.split('\n')
            for line in lines:
                if 'Best Model:' in line:
                    data['next_day_prediction']['best_model'] = line.split(':')[1].strip()
                elif 'Predicted Return:' in line:
                    try:
                        data['next_day_prediction']['predicted_return'] = float(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Raw_DA:' in line:
                    try:
                        data['next_day_prediction']['raw_da'] = float(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Buy_Hold_DA:' in line:
                    try:
                        data['next_day_prediction']['bh_da'] = float(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Recommendation:' in line:
                    data['next_day_prediction']['recommendation'] = line.split(':')[1].strip()

    # Check for available plots
    figures_dir = Path(f'src/reports/{ticker.lower()}_figures')
    if figures_dir.exists():
        plot_files = ['model_comparison.png', 'strategy_performance.png',
                     'prediction_stability.png', 'feature_analysis.png',
                     'next_day_predictions.png']
        for plot_file in plot_files:
            plot_path = figures_dir / plot_file
            if plot_path.exists():
                data['plots'][plot_file.replace('.png', '')] = str(plot_path)

    return data

def create_pdf_report(data: dict, output_path: str) -> None:
    """Create comprehensive PDF report"""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        raise ImportError("ReportLab is required for PDF generation. Install with: pip install reportlab")

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER
    )

    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=18,
        spaceAfter=20,
        alignment=TA_LEFT
    )

    normal_style = styles['Normal']
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ])

    story = []

    # Title Page
    story.append(Paragraph(f"ML Stock Forecasting Report", title_style))
    story.append(Paragraph(f"Ticker: {data['ticker']}", subtitle_style))
    story.append(Paragraph(f"Generated: {data['generation_date']}", normal_style))
    story.append(Spacer(1, 50))

    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['Heading1']))
    story.append(Spacer(1, 12))

    if data['metrics']:
        model_metrics = {k: v for k, v in data['metrics'].items() if k != 'Baseline'}
        if model_metrics:
            best_model = max(model_metrics.keys(),
                            key=lambda x: model_metrics[x].get('Raw_DA', 0))
            best_raw_da = model_metrics[best_model].get('Raw_DA', 0)
            bh_da = data['metrics'].get('Baseline', {}).get('Buy_and_Hold_DA', 0)

            summary_text = f"""
            This report presents a comprehensive analysis of {data['ticker']} stock using advanced machine learning models.
            The best performing model is <b>{best_model}</b> with a raw directional accuracy of <b>{best_raw_da:.1%}</b>
            (vs Buy &amp; Hold baseline of {bh_da:.1%}).
            """

        if data['next_day_prediction']:
            recommendation = data['next_day_prediction'].get('recommendation', 'HOLD')
            pred_return = data['next_day_prediction'].get('predicted_return', 0)
            summary_text += f"""
            <br/><br/>
            <b>Next Day Prediction:</b> {recommendation} (Expected Return: {pred_return:.2%})
            """

        story.append(Paragraph(summary_text, normal_style))
    story.append(Spacer(1, 20))

    # Model Performance Table
    if data['metrics']:
        story.append(Paragraph("Model Performance Metrics", styles['Heading2']))
        story.append(Spacer(1, 12))

        # Prepare table data
        headers = ['Model', 'RMSE', 'MAE', 'Raw DA', 'Confident DA', 'Coverage']
        table_data = [headers]

        for model, metrics in data['metrics'].items():
            if model == 'Baseline':
                row = [
                    'Buy & Hold (baseline)',
                    '—', '—',
                    f"{metrics.get('Buy_and_Hold_DA', 0):.1%}",
                    '—', '100.0%'
                ]
            else:
                row = [
                    model.replace('ML_', ''),
                    f"{metrics.get('RMSE', 0):.4f}",
                    f"{metrics.get('MAE', 0):.4f}",
                    f"{metrics.get('Raw_DA', 0):.1%}",
                    f"{metrics.get('Confident_DA', 0):.1%}",
                    f"{metrics.get('Coverage', 0):.1%}"
                ]
            table_data.append(row)

        # Create table
        table = Table(table_data)
        table.setStyle(table_style)
        story.append(table)
        story.append(Spacer(1, 20))

    # Next Day Prediction Section
    if data['next_day_prediction']:
        story.append(Paragraph("Next Day Trading Recommendation", styles['Heading2']))
        story.append(Spacer(1, 12))

        pred_data = [
            ['Metric', 'Value'],
            ['Best Model', data['next_day_prediction'].get('best_model', 'N/A')],
            ['Predicted Return', f"{data['next_day_prediction'].get('predicted_return', 0) / 100:.2%}"],
            ['Raw DA', f"{data['next_day_prediction'].get('raw_da', 0):.1%}"],
            ['Buy & Hold DA', f"{data['next_day_prediction'].get('bh_da', 0):.1%}"],
            ['Recommendation', data['next_day_prediction'].get('recommendation', 'HOLD')]
        ]

        pred_table = Table(pred_data)
        pred_table.setStyle(table_style)
        story.append(pred_table)
        story.append(Spacer(1, 20))

    # Add page break before charts
    story.append(PageBreak())

    # Charts Section
    story.append(Paragraph("Analysis Charts", styles['Heading1']))
    story.append(Spacer(1, 12))

    chart_descriptions = {
        'model_comparison': 'Model Predictions vs Actual Returns and Error Analysis',
        'strategy_performance': 'Strategy Performance Comparison and Risk Metrics',
        'prediction_stability': 'Prediction Stability and Model Agreement Analysis',
        'feature_analysis': 'Feature Importance and Correlation Analysis'
    }

    for chart_name, chart_path in data['plots'].items():
        if os.path.exists(chart_path):
            story.append(Paragraph(chart_descriptions.get(chart_name, chart_name), styles['Heading2']))
            story.append(Spacer(1, 12))

            # Add image (resize to fit page)
            img = Image(chart_path, width=6*inch, height=4.5*inch)
            story.append(img)
            story.append(Spacer(1, 20))

    # Conclusions
    story.append(PageBreak())
    story.append(Paragraph("Conclusions & Recommendations", styles['Heading1']))
    story.append(Spacer(1, 12))

    # Calculate dynamic DA range
    if data['metrics']:
        model_metrics = {k: v for k, v in data['metrics'].items() if k != 'Baseline'}
        da_values = [metrics.get('Raw_DA', 0) for metrics in model_metrics.values()]
        min_da = min(da_values) if da_values else 0
        max_da = max(da_values) if da_values else 0
        bh_da = data['metrics'].get('Baseline', {}).get('Buy_and_Hold_DA', 0)
    else:
        min_da, max_da, bh_da = 0, 0, 0

    conclusion_text = f"""
    Based on the comprehensive ML analysis of {data['ticker']}, the following conclusions can be drawn:

    1. <b>Model Performance:</b> The machine learning models achieve raw directional accuracies ranging from {min_da:.1%} to {max_da:.1%}, compared to the Buy &amp; Hold baseline of {bh_da:.1%}.

    2. <b>Best Model:</b> {best_model if 'best_model' in locals() else 'XGBoost'} provides the most reliable predictions for trading decisions.

    3. <b>Risk Management:</b> The implemented threshold-based strategy (0.2% threshold) effectively reduces false signals and improves signal quality.

    4. <b>Implementation:</b> Consider implementing the recommended trading strategy with proper position sizing and risk management protocols.

    5. <b>Monitoring:</b> Regular model retraining and performance monitoring is essential for maintaining predictive accuracy.

    <b>Disclaimer:</b> This analysis is for informational purposes only and should not be considered as financial advice.
    """

    story.append(Paragraph(conclusion_text, normal_style))

    # Build PDF
    doc.build(story)
    logging.info(f"PDF report saved to {output_path}")

def create_html_report(data: dict, output_path: str) -> None:
    """Create comprehensive HTML report"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ML Stock Forecasting Report - {data['ticker']}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                text-align: center;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .section {{
                background: white;
                padding: 30px;
                margin-bottom: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .metrics-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .metrics-table th, .metrics-table td {{
                padding: 12px;
                text-align: center;
                border: 1px solid #ddd;
            }}
            .metrics-table th {{
                background-color: #4CAF50;
                color: white;
            }}
            .metrics-table tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .chart-container {{
                text-align: center;
                margin: 30px 0;
            }}
            .chart-container img {{
                max-width: 100%;
                height: auto;
                border: 2px solid #ddd;
                border-radius: 8px;
            }}
            .recommendation {{
                padding: 20px;
                border-left: 5px solid #4CAF50;
                background-color: #e8f5e8;
                margin: 20px 0;
            }}
            .disclaimer {{
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                padding: 15px;
                border-radius: 5px;
                margin-top: 30px;
            }}
            .metric-highlight {{
                font-size: 1.2em;
                font-weight: bold;
                color: #4CAF50;
            }}
            .metrics-table th {{
                cursor: pointer;
                user-select: none;
                position: relative;
                padding-right: 20px;
            }}
            .metrics-table th:hover {{
                background-color: #45a049;
            }}
            .metrics-table th::after {{
                content: '⇅';
                position: absolute;
                right: 6px;
                opacity: 0.4;
                font-size: 0.8em;
            }}
            .metrics-table th.sort-asc::after {{
                content: '↑';
                opacity: 1;
            }}
            .metrics-table th.sort-desc::after {{
                content: '↓';
                opacity: 1;
            }}
            .chart-container img {{
                cursor: zoom-in;
                transition: transform 0.2s;
            }}
            .chart-container img:hover {{
                transform: scale(1.02);
                box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            }}
            .lightbox {{
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.85);
                z-index: 9999;
                justify-content: center;
                align-items: center;
                cursor: zoom-out;
            }}
            .lightbox.active {{
                display: flex;
            }}
            .lightbox img {{
                max-width: 95%;
                max-height: 95%;
                border-radius: 8px;
                box-shadow: 0 0 40px rgba(0,0,0,0.5);
            }}
            .lightbox-close {{
                position: absolute;
                top: 20px;
                right: 30px;
                color: white;
                font-size: 36px;
                cursor: pointer;
                z-index: 10000;
                background: rgba(0,0,0,0.5);
                border-radius: 50%;
                width: 44px;
                height: 44px;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .lightbox-close:hover {{
                background: rgba(255,255,255,0.2);
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 ML Stock Forecasting Report</h1>
            <h2>{data['ticker']} Analysis</h2>
            <p>Generated on: {data['generation_date']}</p>
        </div>

        <div class="section">
            <h2>📋 Trading Strategy Overview</h2>
            <p>This analysis uses a <strong>daily-signal threshold strategy</strong>. Each model produces one prediction per trading day — an expected log-return for the next day. The trading rules are:</p>
            <ul>
                <li><strong>BUY</strong> — if the predicted return exceeds the signal threshold (+0.2%), the model signals to enter a long position at market open and close it at the end of the day.</li>
                <li><strong>HOLD / CASH</strong> — if the predicted return is below the threshold, the model stays out of the market (no position).</li>
                <li><strong>Maximum 1 trade per day</strong> — the model generates exactly one signal per trading day. There is no intraday re-entry or multiple transactions.</li>
            </ul>
            <p><em>Coverage</em> in the metrics table shows what fraction of days the model actually traded (signal exceeded threshold). The remaining days the model sat in cash, avoiding uncertain moves.</p>
        </div>

        <div class="section">
            <h2>🎯 Next Day Trading Recommendation</h2>
    """

    if data['next_day_prediction']:
        recommendation = data['next_day_prediction'].get('recommendation', 'HOLD')
        pred_return = data['next_day_prediction'].get('predicted_return', 0) / 100  # Convert from percentage to decimal
        confidence = data['next_day_prediction'].get('raw_da', 0)
        bh_da = data['next_day_prediction'].get('bh_da', 0)

        html_content += f"""
            <div class="recommendation">
                <p><strong>Action:</strong> {recommendation}</p>
                <p><strong>Expected Return:</strong> {pred_return:.2%}</p>
                <p><strong>Raw DA:</strong> {confidence:.1%} (Buy & Hold baseline: {bh_da:.1%})</p>
            </div>
        """

    # Add Next Day Prediction Chart
    if 'next_day_predictions' in data['plots']:
        chart_path = data['plots']['next_day_predictions']
        if os.path.exists(chart_path):
            # Convert image to base64 for embedding
            with open(chart_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')

            html_content += f"""
        <div class="section">
            <h2>📊 Next Day Price Predictions & Recommendations</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{img_data}" alt="next_day_predictions">
            </div>
        </div>
            """

    html_content += """
        <div class="section">
            <h2>📊 Executive Summary</h2>
    """

    if data['metrics']:
        model_metrics_html = {k: v for k, v in data['metrics'].items() if k != 'Baseline'}
        if model_metrics_html:
            best_model_html = max(model_metrics_html.keys(),
                             key=lambda x: model_metrics_html[x].get('Raw_DA', 0))
            best_raw_da = model_metrics_html[best_model_html].get('Raw_DA', 0)
            bh_da_html = data['metrics'].get('Baseline', {}).get('Buy_and_Hold_DA', 0)

            html_content += f"""
                <p>This comprehensive ML analysis of <strong>{data['ticker']}</strong> achieves
                a best raw directional accuracy of <span class="metric-highlight">{best_raw_da:.1%}</span>
                (vs Buy &amp; Hold baseline of {bh_da_html:.1%}).</p>
            """

    html_content += """
        </div>

        <div class="section">
            <h2>📈 Model Performance Metrics</h2>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>RMSE</th>
                        <th>MAE</th>
                        <th>Raw DA</th>
                        <th>Confident DA</th>
                        <th>Coverage</th>
                        <th>Trades</th>
                    </tr>
                </thead>
                <tbody>
    """

    if data['metrics']:
        for model, metrics in data['metrics'].items():
            if model == 'Baseline':
                bh_val = metrics.get('Buy_and_Hold_DA', 0)
                html_content += f"""
                    <tr style="background-color: #e8e8e8; font-style: italic;">
                        <td>Buy &amp; Hold (baseline)</td>
                        <td>—</td>
                        <td>—</td>
                        <td>{bh_val:.1%}</td>
                        <td>—</td>
                        <td>100.0%</td>
                        <td>—</td>
                    </tr>
                """
            else:
                total_days = int(metrics.get('Total_Test_Days', 0))
                coverage = metrics.get('Coverage', 0)
                trades = int(round(coverage * total_days))
                trades_str = f"{trades} / {total_days}" if total_days > 0 else "—"
                html_content += f"""
                    <tr>
                        <td>{model.replace('ML_', '')}</td>
                        <td>{metrics.get('RMSE', 0):.4f}</td>
                        <td>{metrics.get('MAE', 0):.4f}</td>
                        <td>{metrics.get('Raw_DA', 0):.1%}</td>
                        <td>{metrics.get('Confident_DA', 0):.1%}</td>
                        <td>{coverage:.1%}</td>
                        <td>{trades_str}</td>
                    </tr>
                """

    html_content += """
                </tbody>
            </table>

            <div style="margin-top: 20px; padding: 15px; background-color: #f0f4ff; border-radius: 8px; font-size: 0.9em;">
                <h4 style="margin-top: 0;">📖 Metric Descriptions</h4>
                <ul style="margin-bottom: 0;">
                    <li><strong>RMSE</strong> (Root Mean Squared Error) — average magnitude of prediction errors, penalizing larger errors more heavily. Lower is better.</li>
                    <li><strong>MAE</strong> (Mean Absolute Error) — average absolute difference between predicted and actual log-returns. Lower is better.</li>
                    <li><strong>Raw DA</strong> (Raw Directional Accuracy) — percentage of days where the model correctly predicted the direction of price movement (up/down), calculated on <em>all</em> trading days without any filtering.</li>
                    <li><strong>Confident DA</strong> (High-Confidence Directional Accuracy) — directional accuracy calculated only on days where the model's predicted return exceeded the signal threshold (&plusmn;0.2%). Represents the trading strategy accuracy — the model trades only when confident.</li>
                    <li><strong>Coverage</strong> — fraction of trading days where the model generates a trading signal (|prediction| &gt; threshold). Higher coverage = more frequent trading.</li>
                    <li><strong>Trades</strong> — absolute number of days the model traded out of total test days (e.g. "228 / 251" means the model traded on 228 out of 251 available days).</li>
                    <li><strong>Buy &amp; Hold (baseline)</strong> — directional accuracy of a naive strategy that always predicts "price goes up". Equals the percentage of days the market actually rose. Models should exceed this to demonstrate real predictive power.</li>
                </ul>
            </div>
        </div>
    """

    # Add charts (excluding next_day_predictions which is shown earlier)
    chart_titles = {
        'model_comparison': 'Model Comparison & Error Analysis',
        'strategy_performance': 'Strategy Performance Analysis',
        'prediction_stability': 'Prediction Stability & Agreement',
        'feature_analysis': 'Feature Analysis & Correlations'
    }

    for chart_name, chart_path in data['plots'].items():
        if chart_name != 'next_day_predictions' and os.path.exists(chart_path):
            # Convert image to base64 for embedding
            with open(chart_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')

            html_content += f"""
        <div class="section">
            <h2>📊 {chart_titles.get(chart_name, chart_name.replace('_', ' ').title())}</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{img_data}" alt="{chart_name}">
            </div>
        </div>
            """

    # Calculate dynamic DA range for HTML
    if data['metrics']:
        model_metrics_concl = {k: v for k, v in data['metrics'].items() if k != 'Baseline'}
        da_values = [metrics.get('Raw_DA', 0) for metrics in model_metrics_concl.values()]
        min_da_html = min(da_values) if da_values else 0
        max_da_html = max(da_values) if da_values else 0
        bh_da_concl = data['metrics'].get('Baseline', {}).get('Buy_and_Hold_DA', 0)
    else:
        min_da_html, max_da_html, bh_da_concl = 0, 0, 0

    # Conclusions
    html_content += f"""
        <div class="section">
            <h2>🎯 Conclusions & Recommendations</h2>
            <ul>
                <li><strong>Model Performance:</strong> ML models achieve raw directional accuracies from {min_da_html:.1%} to {max_da_html:.1%} (Buy &amp; Hold baseline: {bh_da_concl:.1%})</li>
                <li><strong>Risk Management:</strong> Threshold-based strategy (0.2% threshold) effectively reduces false signals and improves signal quality</li>
                <li><strong>Implementation:</strong> Consider implementing the recommended trading strategy with proper position sizing and risk management protocols</li>
                <li><strong>Monitoring:</strong> Regular model retraining and performance monitoring is essential for maintaining predictive accuracy</li>
                <li><strong>Next Steps:</strong> Focus on the best performing model (XGBoost) for production deployment with continuous validation</li>
            </ul>
        </div>

        <div class="disclaimer">
            <h3>⚠️ Disclaimer</h3>
            <p>This analysis is for informational purposes only and should not be considered as financial advice.
            Past performance does not guarantee future results. Always conduct your own research and consult with
            qualified financial advisors before making investment decisions.</p>
        </div>
        <div class="lightbox" id="lightbox">
            <span class="lightbox-close" id="lightbox-close">&times;</span>
            <img id="lightbox-img" src="" alt="">
        </div>

        <script>
        // Sortable table
        document.querySelectorAll('.metrics-table th').forEach(function(th) {{
            th.addEventListener('click', function() {{
                var table = th.closest('table');
                var tbody = table.querySelector('tbody');
                var rows = Array.from(tbody.querySelectorAll('tr'));
                var colIndex = Array.from(th.parentNode.children).indexOf(th);
                var isAsc = th.classList.contains('sort-asc');

                table.querySelectorAll('th').forEach(function(h) {{
                    h.classList.remove('sort-asc', 'sort-desc');
                }});

                rows.sort(function(a, b) {{
                    var aText = a.children[colIndex].textContent.trim();
                    var bText = b.children[colIndex].textContent.trim();

                    // Parse numbers: handle percentages, fractions, dashes
                    var aNum = parseFloat(aText.replace('%', '').split('/')[0]);
                    var bNum = parseFloat(bText.replace('%', '').split('/')[0]);

                    if (!isNaN(aNum) && !isNaN(bNum)) {{
                        return isAsc ? bNum - aNum : aNum - bNum;
                    }}
                    return isAsc ? bText.localeCompare(aText) : aText.localeCompare(bText);
                }});

                th.classList.add(isAsc ? 'sort-desc' : 'sort-asc');
                rows.forEach(function(row) {{ tbody.appendChild(row); }});
            }});
        }});

        // Image lightbox
        var lightbox = document.getElementById('lightbox');
        var lightboxImg = document.getElementById('lightbox-img');

        document.querySelectorAll('.chart-container img').forEach(function(img) {{
            img.addEventListener('click', function() {{
                lightboxImg.src = img.src;
                lightbox.classList.add('active');
            }});
        }});

        lightbox.addEventListener('click', function() {{
            lightbox.classList.remove('active');
        }});

        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') lightbox.classList.remove('active');
        }});
        </script>
    </body>
    </html>
    """

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logging.info(f"HTML report saved to {output_path}")

def main():
    """Main function"""
    setup_logging()

    parser = argparse.ArgumentParser(description='Generate comprehensive ML analysis report')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker (default: AAPL)')
    parser.add_argument('--format', type=str, choices=['pdf', 'html', 'both'], default='both',
                       help='Report format (default: both)')
    parser.add_argument('--output-dir', type=str, default='reports',
                       help='Output directory (default: reports)')

    args = parser.parse_args()

    logging.info(f"Generating {args.format} report for {args.ticker}")

    # Load all data
    data = load_metrics_data(args.ticker)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Check PDF availability
    pdf_available = check_pdf_availability()

    # Generate reports
    if args.format in ['pdf', 'both']:
        if pdf_available:
            pdf_path = os.path.join(args.output_dir, f'{args.ticker.lower()}_ml_report.pdf')
            create_pdf_report(data, pdf_path)
        else:
            logging.warning("PDF generation requested but ReportLab not available. Install with: pip install reportlab")
            if args.format == 'pdf':
                logging.info("Falling back to HTML generation only")
                args.format = 'html'

    if args.format in ['html', 'both']:
        html_path = os.path.join(args.output_dir, f'{args.ticker.lower()}_ml_report.html')
        create_html_report(data, html_path)

    logging.info("Report generation completed successfully!")

if __name__ == "__main__":
    main()