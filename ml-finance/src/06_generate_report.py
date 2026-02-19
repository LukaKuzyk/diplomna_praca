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
                elif line.endswith('_Returns:') or line.endswith('_Probability:'):
                    current_model = line.replace(':', '')
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
                     'next_day_predictions.png', 'next_day_predictions_clf.png']
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
    story.append(Paragraph(f"ML Report Predikcie Akcií", title_style))
    story.append(Paragraph(f"Ticker: {data['ticker']}", subtitle_style))
    story.append(Paragraph(f"Vygenerované: {data['generation_date']}", normal_style))
    story.append(Spacer(1, 50))

    # Executive Summary
    story.append(Paragraph("Manažérske Zhrnutie (Executive Summary)", styles['Heading1']))
    story.append(Spacer(1, 12))

    if data['metrics']:
        model_metrics = {k: v for k, v in data['metrics'].items() if k != 'Baseline'}
        if model_metrics:
            best_model = max(model_metrics.keys(),
                            key=lambda x: model_metrics[x].get('Raw_DA', 0))
            best_raw_da = model_metrics[best_model].get('Raw_DA', 0)
            bh_da = data['metrics'].get('Baseline', {}).get('Buy_and_Hold_DA', 0)

            summary_text = f"""
            Tento report predstavuje komplexnú analýzu akcie {data['ticker']} pomocou pokročilých modelov strojového učenia (ML).
            Najvýkonnejší model je <b>{best_model}</b> s raw smerovou presnosťou (Raw DA) <b>{best_raw_da:.1%}</b>
            (oproti Buy &amp; Hold baseline {bh_da:.1%}).
            """

        if data['next_day_prediction']:
            recommendation = data['next_day_prediction'].get('recommendation', 'HOLD')
            pred_return = data['next_day_prediction'].get('predicted_return', 0)
            summary_text += f"""
            <br/><br/>
            <b>Predikcia na Ďalší Deň:</b> {recommendation} (Očakávaný Výnos: {pred_return:.2%})
            """

        story.append(Paragraph(summary_text, normal_style))
    story.append(Spacer(1, 20))

    # Model Performance Tables
    if data['metrics']:
        story.append(Paragraph("Metriky Výkonnosti Modelov - Regresia", styles['Heading2']))
        story.append(Spacer(1, 12))

        headers = ['Model', 'RMSE', 'MAE', 'Raw DA', 'Confident DA', 'Coverage']
        table_data = [headers]

        reg_models = [(m, mets) for m, mets in data['metrics'].items() if m.startswith('ML_REG_')]
        reg_models.sort(key=lambda x: x[1].get('Confident_DA', 0), reverse=True)

        baseline_metrics = data['metrics'].get('Baseline', {})
        if baseline_metrics:
            row = [
                'Buy & Hold',
                '—', '—',
                f"{baseline_metrics.get('Buy_and_Hold_DA', 0):.1%}",
                '—', '100.0%'
            ]
            table_data.append(row)

        for model, metrics in reg_models:
            row = [
                model.replace('ML_REG_', '').replace('_Returns', ''),
                f"{metrics.get('RMSE', 0):.4f}",
                f"{metrics.get('MAE', 0):.4f}",
                f"{metrics.get('Raw_DA', 0):.1%}",
                f"{metrics.get('Confident_DA', 0):.1%}",
                f"{metrics.get('Coverage', 0):.1%}"
            ]
            table_data.append(row)

        table = Table(table_data)
        table.setStyle(table_style)
        story.append(table)
        story.append(Spacer(1, 20))
        
        story.append(Paragraph("Metriky Výkonnosti Modelov - Klasifikácia", styles['Heading2']))
        story.append(Spacer(1, 12))

        headers_cl = ['Model', 'Mean Prob', 'Raw DA', 'Conf DA (>55%)', 'Coverage']
        table_data_cl = [headers_cl]

        cl_models = [(m, mets) for m, mets in data['metrics'].items() if m.startswith('ML_CL_')]
        cl_models.sort(key=lambda x: x[1].get('Confident_DA', 0), reverse=True)

        if baseline_metrics:
            row = [
                'Buy & Hold', '—',
                f"{baseline_metrics.get('Buy_and_Hold_DA', 0):.1%}",
                f"{baseline_metrics.get('Buy_and_Hold_DA', 0):.1%}", '100.0%'
            ]
            table_data_cl.append(row)

        for model, metrics in cl_models:
            row = [
                model.replace('ML_CL_', '').replace('_Probability', ''),
                f"{metrics.get('Mean_Probability', 0):.2%}",
                f"{metrics.get('Raw_DA', 0):.1%}",
                f"{metrics.get('Confident_DA', 0):.1%}",
                f"{metrics.get('Coverage', 0):.1%}"
            ]
            table_data_cl.append(row)

        table_cl = Table(table_data_cl)
        table_cl.setStyle(table_style)
        story.append(table_cl)
        story.append(Spacer(1, 20))

    # Next Day Prediction Section
    if data['next_day_prediction']:
        story.append(Paragraph("Odporúčanie pre Zajtrajšie Obchodovanie", styles['Heading2']))
        story.append(Spacer(1, 12))

        pred_data = [
            ['Metric (Metrika)', 'Value (Hodnota)'],
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
    story.append(Paragraph("Analytické Grafy (Analysis Charts)", styles['Heading1']))
    story.append(Spacer(1, 12))

    chart_descriptions = {
        'model_comparison': 'Porovnanie Modelov & Analýza Chýb (Model Predictions vs Actual Returns)',
        'strategy_performance': 'Výkonnosť Stratégie & Metriky Rizika (Strategy Performance)',
        'prediction_stability': 'Stabilita Predikcií & Zhoda Modelov (Prediction Stability)',
        'feature_analysis': 'Analýza Atribútov & Korelácie (Feature Importance)'
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
    story.append(Paragraph("Závery & Odporúčania (Conclusions)", styles['Heading1']))
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
    Na základe komplexnej ML analýzy akcie {data['ticker']} je možné vyvodiť nasledujúce závery:

    1. <b>Výkonnosť Modelov:</b> Modely strojového učenia dosahujú raw smerovú presnosť (Raw DA) v rozsahu od {min_da:.1%} do {max_da:.1%}, v porovnaní s Buy &amp; Hold baseline {bh_da:.1%}.

    2. <b>Najlepší Model:</b> {best_model if 'best_model' in locals() else 'XGBoost'} poskytuje najspoľahlivejšie predikcie pre obchodné rozhodnutia.

    3. <b>Riadenie Rizík:</b> Implementovaná stratégia založená na prahu (0.2%) efektívne redukuje falošné signály a zlepšuje kvalitu signálov.

    4. <b>Implementácia:</b> Zvážte implementáciu odporúčanej obchodnej stratégie s primeraným dimenzovaním pozícií a protokolmi riadenia rizík.

    5. <b>Monitorovanie:</b> Pravidelné pretrénovanie modelov a monitorovanie výkonnosti je nevyhnutné pre udržanie presnosti predikcií.

    <b>Vylúčenie Zodpovednosti (Disclaimer):</b> Táto analýza slúži len na informačné účely a nemala by byť považovaná za finančné poradenstvo.
    """

    story.append(Paragraph(conclusion_text, normal_style))

    # Build PDF
    doc.build(story)
    logging.info(f"PDF report saved to {output_path}")

def create_html_report(data: dict, output_path: str) -> None:
    """Create comprehensive HTML report"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="sk">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ML Report Predikcie Akcií - {data['ticker']}</title>
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
            <h1>🤖 ML Report Predikcie Akcií</h1>
            <h2>Analýza pre {data['ticker']}</h2>
            <p>Vygenerované dňa: {data['generation_date']}</p>
        </div>

        <div class="section">
            <h2>📋 Trading Strategy Overview </h2>
            <p>Táto analýza využíva <strong>daily-signal threshold strategy</strong> (stratégiu denného prahu signálu). Každý model generuje jednu predikciu na obchodný deň — očakávaný log-výnos na nasledujúci deň. Pravidlá obchodovania sú:</p>
            <ul>
                <li><strong>BUY</strong> — ak predikovaný výnos prekročí prah signálu (+0.2%), model signalizuje otvorenie long pozície pri otvorení trhu a jej uzavretie na konci dňa.</li>
                <li><strong>HOLD / CASH</strong> — ak je predikovaný výnos pod prahom, model zostáva mimo trhu (žiadna pozícia).</li>
                <li><strong>Maximálne 1 obchod denne</strong> — model generuje presne jeden signál na obchodný deň. Žiadne intraday vstupy alebo viacnásobné transakcie.</li>
            </ul>
            <p><em>Coverage</em> v tabuľke metrík ukazuje, v akej časti dní model skutočne obchodoval (signál prekročil prah). Zvyšné dni model zostal v hotovosti, vyhýbajúc sa neistým pohybom.</p>
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

    # Add Next Day Prediction Chart(s)
    if 'next_day_predictions' in data['plots'] or 'next_day_predictions_clf' in data['plots']:
        html_content += f"""
        <div class="section">
            <h2>📊 Next Day Price Predictions & Recommendations</h2>
        """
        if 'next_day_predictions' in data['plots']:
            chart_path = data['plots']['next_day_predictions']
            if os.path.exists(chart_path):
                with open(chart_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                html_content += f"""
            <div class="chart-container">
                <img src="data:image/png;base64,{img_data}" alt="next_day_predictions">
            </div>
                """
        
        if 'next_day_predictions_clf' in data['plots']:
            chart_path_clf = data['plots']['next_day_predictions_clf']
            if os.path.exists(chart_path_clf):
                with open(chart_path_clf, "rb") as img_file:
                    img_data_clf = base64.b64encode(img_file.read()).decode('utf-8')
                html_content += f"""
            <div class="chart-container" style="margin-top: 40px;">
                <img src="data:image/png;base64,{img_data_clf}" alt="next_day_predictions_clf">
            </div>
                """
                
        html_content += """
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
                <p>Táto komplexná ML analýza akcie <strong>{data['ticker']}</strong> dosahuje
                najlepšiu raw smerovú presnosť (Raw DA) <span class="metric-highlight">{best_raw_da:.1%}</span>
                (oproti Buy &amp; Hold baseline {bh_da_html:.1%}).</p>
            """

    html_content += """
        </div>

        <div class="section">
            <h2>📈 Regression Model Performance</h2>
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
        baseline_metrics = data['metrics'].get('Baseline', {})
        if baseline_metrics:
            bh_val = baseline_metrics.get('Buy_and_Hold_DA', 0)
            html_content += f"""
                <tr style="background-color: #e8e8e8; font-style: italic;">
                    <td>Buy &amp; Hold</td>
                    <td>—</td>
                    <td>—</td>
                    <td>{bh_val:.1%}</td>
                    <td>{bh_val:.1%}</td>
                    <td>100.0%</td>
                    <td></td>
                </tr>
            """
            
        reg_models = [(m, mets) for m, mets in data['metrics'].items() if m.startswith('ML_REG_')]
        reg_models.sort(key=lambda x: x[1].get('Confident_DA', 0), reverse=True)
        
        for model, metrics in reg_models:
            total_days = int(metrics.get('Total_Test_Days', 0))
            coverage = metrics.get('Coverage', 0)
            trades = int(round(coverage * total_days))
            trades_str = f"{trades} / {total_days}" if total_days > 0 else "—"
            
            # e.g. ML_REG_RF_Returns -> RF
            model_display = model.replace('ML_REG_', '').replace('_Returns', '')
            html_content += f"""
                <tr>
                    <td>{model_display} (Reg)</td>
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
                <h4 style="margin-top: 0;">📖 Popis Metrík (Regresia)</h4>
                <ul style="margin-bottom: 0;">
                    <li><strong>RMSE</strong> (Root Mean Squared Error) — priemerná veľkosť chýb predikcie, penalizuje väčšie chyby výraznejšie. Čím nižšie, tým lepšie.</li>
                    <li><strong>MAE</strong> (Mean Absolute Error) — priemerný absolútny rozdiel medzi predikovanými a skutočnými log-výnosmi. Čím nižšie, tým lepšie.</li>
                    <li><strong>Raw DA</strong> (Raw Directional Accuracy) — percento dní, kedy model správne predikoval smer pohybu ceny (hore/dole), počítané na <em>všetkých</em> obchodných dňoch bez filtrovania.</li>
                    <li><strong>Confident DA</strong> (High-Confidence Directional Accuracy) — smerová presnosť počítaná len v dňoch, kedy predikovaný výnos modelu prekročil prah signálu (&plusmn;0.2%). Reprezentuje presnosť obchodnej stratégie — model obchoduje len keď je si istý.</li>
                    <li><strong>Coverage</strong> — podiel obchodných dní, kedy model generuje obchodný signál (|prediction| &gt; prah). Vyššie coverage = častejšie obchodovanie.</li>
                    <li><strong>Trades</strong> — absolútny počet dní, kedy model obchodoval z celkového počtu testovacích dní (napr. "228 / 251" znamená, že model obchodoval 228 z 251 dostupných dní).</li>
                    <li><strong>Buy &amp; Hold</strong> — smerová presnosť naivnej stratégie, ktorá vždy predikuje "cena pôjde hore". Rovná sa percentu dní, kedy trh skutočne rástol. Modely by mali túto hodnotu prekonať, aby preukázali skutočnú predikčnú schopnosť.</li>
                </ul>
            </div>
            
        </div>
    """
    html_content += """
                </tbody>
            </table>
            
            <h2 style="margin-top: 40px;">🎯 Classification Model Performance</h2>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Mean Probability</th>
                        <th>Raw DA</th>
                        <th>Confident DA (>55%)</th>
                        <th>Coverage</th>
                        <th>Trades</th>
                    </tr>
                </thead>
                <tbody>
    """
    if data['metrics']:
        # Also print Baseline row for classifiers
        baseline_metrics = data['metrics'].get('Baseline', {})
        if baseline_metrics:
            bh_val = baseline_metrics.get('Buy_and_Hold_DA', 0)
            html_content += f"""
                <tr style="background-color: #e8e8e8; font-style: italic;">
                    <td>Buy &amp; Hold</td>
                    <td>—</td>
                    <td>{bh_val:.1%}</td>
                    <td>{bh_val:.1%}</td>
                    <td>100.0%</td>
                    <td></td>
                </tr>
            """
            
        cl_models = [(m, mets) for m, mets in data['metrics'].items() if m.startswith('ML_CL_')]
        cl_models.sort(key=lambda x: x[1].get('Confident_DA', 0), reverse=True)
        
        for model, metrics in cl_models:
            total_days = int(metrics.get('Total_Test_Days', 0))
            coverage = metrics.get('Coverage', 0)
            trades = int(round(coverage * total_days))
            trades_str = f"{trades} / {total_days}" if total_days > 0 else "—"
            
            # e.g. ML_CL_RF_Probability -> RF
            model_display = model.replace('ML_CL_', '').replace('_Probability', '')
            html_content += f"""
                <tr>
                    <td>{model_display} (Clf)</td>
                    <td>{metrics.get('Mean_Probability', 0):.2%}</td>
                    <td>{metrics.get('Raw_DA', 0):.1%}</td>
                    <td>{metrics.get('Confident_DA', 0):.1%}</td>
                    <td>{coverage:.1%}</td>
                    <td>{trades_str}</td>
                </tr>
            """
    html_content += """
                </tbody>
            </table>
            
            <div style="margin-top: 20px; padding: 15px; background-color: #fff0f5; border-radius: 8px; font-size: 0.9em;">
                <h4 style="margin-top: 0;">📖 Popis Metrík (Klasifikácia)</h4>
                <ul style="margin-bottom: 0;">
                    <li><strong>Mean Probability</strong> — priemerná pravdepodobnosť z predikcií modelu. Hodnota blízko 50% signalizuje neistotu, kým vychýlené hodnoty indikujú silnejšie trendy.</li>
                    <li><strong>Raw DA</strong> — percento dní, kedy model správne predikoval smer (pravdepodobnosť > 50% = výnos nahor), merané plošne na všetkých dňoch.</li>
                    <li><strong>Confident DA (>55%)</strong> — smerová presnosť počítaná len v dňoch, kedy si model bol viac istý pohnutím trhu (t.j. predpovedal P(Up) > 55% pre rastové signály, alebo P(Up) < 45% pre klesajúce). Týmto odfiltruje neutrálne odhady blízko 50%.</li>
                    <li><strong>Coverage</strong> — percentuálny podiel dní, kedy model vyprodukoval silný "confident" signál (pravdepodobnosť vychýlená aspoň o 5% od nezávislých 50%).</li>
                    <li><strong>Trades</strong> — skutočný počet vygenerovaných sebavedomých obchodných rozhodnutí.</li>
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

    feature_descriptions_html = """
            <div style="margin-bottom: 20px; font-size: 0.92em; line-height: 1.6;">
                <h3 style="margin-bottom: 10px;">Prehľad použitých prediktorov</h3>
                <table style="width:100%; border-collapse: collapse; font-size: 0.9em;">
                    <tr style="background: #f0f0f0;">
                        <th style="text-align:left; padding: 6px 10px; border-bottom: 2px solid #ddd;">Kategória</th>
                        <th style="text-align:left; padding: 6px 10px; border-bottom: 2px solid #ddd;">Featury</th>
                        <th style="text-align:left; padding: 6px 10px; border-bottom: 2px solid #ddd;">Popis</th>
                    </tr>
                    <tr>
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top;"><strong>Return Lags</strong></td>
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top;"><code>log_ret_lag_1..30</code></td>
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee;">Logaritmické výnosy oneskorené o 1–30 dní. Zachytávajú autokoreláciu a momentum v cenových pohyboch.</td>
                    </tr>
                    <tr style="background: #fafafa;">
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top;"><strong>Volume</strong></td>
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top;"><code>volume</code>, <code>volume_lag_1..5</code>, <code>volume_ma_5/20</code></td>
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee;">Objem obchodovania a jeho kĺzavé priemery. Vysoký objem potvrdzuje silu trendu, nízky signalizuje neistotu.</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top;"><strong>Technical</strong></td>
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top;"><code>sma_5/20</code>, <code>rsi_14</code>, <code>macd</code>, <code>bb_upper/lower/middle</code>, <code>stoch_k/d</code>, <code>atr_14</code>, <code>cci_20</code>, <code>momentum_5/10</code>, <code>volatility</code></td>
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee;">Technické indikátory: kĺzavé priemery (SMA), index relatívnej sily (RSI), MACD, Bollinger Bands, stochastic oscilátor, ATR (priemerný rozsah), CCI, momentum a volatilita.</td>
                    </tr>
                    <tr style="background: #fafafa;">
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top;"><strong>Statistical</strong></td>
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top;"><code>rolling_skew_20</code>, <code>rolling_kurt_20</code></td>
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee;">Šikmosť a špicatosť výnosov za 20 dní. Zachytávajú asymetriu a extrémne pohyby v distribúcii výnosov.</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top;"><strong>Calendar</strong></td>
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top;"><code>day_of_week</code>, <code>month</code></td>
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee;">Deň v týždni a mesiac. Zachytávajú sezónne vzory (napr. „pondelkový efekt\", január efekt).</td>
                    </tr>
                    <tr style="background: #fafafa;">
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top;"><strong>Market</strong></td>
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top;"><code>vix_close</code>, <code>vix_change</code>, <code>qqq_change</code>, <code>snp500_change</code> + lag 1–3</td>
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee;">VIX (index strachu) — meria očakávanú volatilitu trhu. QQQ — výkonnosť technologického sektora (Nasdaq-100 ETF). S&amp;P 500 — zmena širokého trhového indexu. Lagy zachytávajú oneskorenú reakciu.</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top;"><strong>Earnings</strong></td>
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top;"><code>earnings_week</code></td>
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee;">Binárny príznak — 1, ak sa v najbližších 7 dňoch očakáva zverejnenie kvartálnych výsledkov spoločnosti.</td>
                    </tr>
                    <tr style="background: #fafafa;">
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top;"><strong>Search Trends</strong></td>
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top;"><code>iphone_search</code>, <code>ai_search</code>, <code>election_search</code>, <code>trump_search</code>, <code>stock_search</code> + lag 1–3</td>
                        <td style="padding: 6px 10px; border-bottom: 1px solid #eee;">Google Trends — týždenný záujem o kľúčové témy. Lagy zachytávajú oneskorený vplyv verejného záujmu na trh.</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 10px; vertical-align: top;"><strong>News Trends</strong></td>
                        <td style="padding: 6px 10px; vertical-align: top;"><code>war_news</code>, <code>unemployment_news</code>, <code>tariffs_news</code>, <code>earnings_news</code>, <code>ai_news</code> + lag 1–3</td>
                        <td style="padding: 6px 10px;">Google News Trends — frekvencia spravodajských článkov na kľúčové témy. Odrážajú mediálnu náladu a sentiment.</td>
                    </tr>
                </table>
            </div>
    """

    for chart_name, chart_path in data['plots'].items():
        if chart_name not in ['next_day_predictions', 'next_day_predictions_clf'] and os.path.exists(chart_path):
            # Convert image to base64 for embedding
            with open(chart_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')

            extra_content = feature_descriptions_html if chart_name == 'feature_analysis' else ''

            html_content += f"""
        <div class="section">
            <h2>📊 {chart_titles.get(chart_name, chart_name.replace('_', ' ').title())}</h2>
            {extra_content}
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
            <h2>🎯 Závery</h2>
            <ul>
                <li><strong>Výkonnosť Modelov:</strong> ML modely dosahujú raw smerovú presnosť od {min_da_html:.1%} do {max_da_html:.1%} (Buy &amp; Hold baseline: {bh_da_concl:.1%})</li>
                <li><strong>Riadenie Rizík:</strong> Stratégia založená na prahu (0.2%) efektívne redukuje falošné signály a zlepšuje kvalitu signálov</li>
            </ul>
        </div>

        <div class="disclaimer">
            <h3>⚠️ Disclaimer</h3>
            <p>Táto analýza slúži len na informačné účely a nemala by byť považovaná za finančné poradenstvo.
            Minulá výkonnosť nie je zárukou budúcich výsledkov. Pred investičným rozhodnutím vždy vykonajte vlastný
            prieskum a poraďte sa s kvalifikovanými finančnými poradcami.</p>
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

        // Auto-sort tables by Confident DA on load
        window.addEventListener('DOMContentLoaded', function() {{
            document.querySelectorAll('.metrics-table').forEach(function(table) {{
                var ths = Array.from(table.querySelectorAll('th'));
                var confDaTh = ths.find(function(t) {{ return t.textContent.indexOf('Confident DA') !== -1; }});
                if (confDaTh) {{
                    confDaTh.classList.add('sort-asc'); // next click makes it desc
                    confDaTh.click();
                }}
            }});
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
    parser.add_argument('--format', type=str, choices=['pdf', 'html', 'both'], default='html',
                       help='Report format (default: html)')
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