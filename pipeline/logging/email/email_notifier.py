import smtplib
import ssl
import os
from pathlib import Path
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import json
import traceback
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

env_path = Path(__file__).resolve().parent.parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv(override=True)

class ExperimentEmailNotifier:
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        
        self.sender_email = os.environ.get("EMAIL_SENDER")
        self.sender_password = os.environ.get("EMAIL_PASSWORD")
        
        if not self.sender_email or not self.sender_password:
            raise ValueError("Email credentials not provided. Set EMAIL_SENDER and EMAIL_PASSWORD environment variables.")
        
        recipients_str = os.environ.get("EMAIL_RECIPIENTS", self.sender_email)
        if recipients_str:
            self.recipient_emails = [email.strip() for email in recipients_str.split(',')]
        else:
            self.recipient_emails = [self.sender_email]
    
    def send_experiment_notification(
        self,
        experiment_name: str,
        results: Dict[str, Any],
        duration: float,
        status: str = "completed",
        error_message: str = None,
        attach_plots: bool = True
    ):
        try:
            is_componentwise = self._is_componentwise_results(results)
            
            subject = self._create_subject(experiment_name, status, is_componentwise)
            
            message = MIMEMultipart("mixed")
            message["Subject"] = subject
            message["From"] = self.sender_email
            message["To"] = ", ".join(self.recipient_emails)
            
            html_part = MIMEMultipart("related")
            
            if is_componentwise:
                html_content = self._create_componentwise_html_content(
                    experiment_name, results, duration, status, error_message, attach_plots
                )
            else:
                html_content = self._create_html_content(
                    experiment_name, results, duration, status, error_message, attach_plots
                )
            
            html_part.attach(MIMEText(html_content, "html"))
            
            if attach_plots and status == "completed" and results:
                if is_componentwise:
                    self._attach_componentwise_plots(html_part, results)
                else:
                    self._attach_inline_plots(html_part, results)
            
            message.attach(html_part)
            
            self._send_email(message)
            print(f"✅ Email notification sent successfully to {', '.join(self.recipient_emails)}")
            
        except Exception as e:
            print(f"❌ Failed to send email notification: {e}")
            traceback.print_exc()
    
    def _is_componentwise_results(self, results: Dict[str, Any]) -> bool:
        if not results:
            return False
        return 'component_results' in results and 'component_order' in results
    
    def _create_subject(self, experiment_name: str, status: str, is_componentwise: bool = False) -> str:
        status_emoji = {
            "completed": "✅",
            "failed": "❌",
            "early_stopped": "⚡"
        }
        emoji = status_emoji.get(status, "📊")
        mode = "Component-wise" if is_componentwise else "Global"
        return f"{emoji} {mode} RAG Optimization - {experiment_name} - {status.upper()}"
    
    def _create_componentwise_html_content(
        self,
        experiment_name: str,
        results: Dict[str, Any],
        duration: float,
        status: str,
        error_message: str,
        attach_plots: bool = True
    ) -> str:
        status_color = {
            "completed": "#28a745",
            "failed": "#dc3545",
            "early_stopped": "#ffc107"
        }
        color = status_color.get(status, "#17a2b8")
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: {color}; color: white; padding: 20px; border-radius: 5px; }}
                .content {{ margin-top: 20px; }}
                .component-card {{ background-color: #f8f9fa; padding: 15px; margin: 15px 0; border-radius: 5px; border-left: 4px solid #007bff; }}
                .component-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
                .component-name {{ font-size: 18px; font-weight: bold; color: #333; }}
                .component-score {{ font-size: 20px; font-weight: bold; color: #28a745; }}
                .config-box {{ background-color: #e9ecef; padding: 10px; margin: 10px 0; border-radius: 5px; font-family: monospace; font-size: 12px; }}
                .summary-table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                .summary-table th, .summary-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .summary-table th {{ background-color: #f2f2f2; }}
                .plot-container {{ margin: 20px 0; text-align: center; }}
                .plot-container img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Component-wise RAG Pipeline Optimization</h1>
                <h2>{experiment_name}</h2>
                <p>Status: <strong>{status.upper()}</strong> | Duration: {self._format_duration(duration)}</p>
                <p>Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="content">
        """
        
        if status == "failed" and error_message:
            html += f"""
                <div style="background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <strong>Error:</strong> {error_message}
                </div>
            """
        
        if results and status == "completed":
            html += self._create_componentwise_summary(results)
            html += self._create_component_details(results)
            
            if attach_plots:
                html += """
                <h2>Optimization Progress</h2>
                <div class="plot-container">
                    <img src="cid:component_scores" alt="Component Scores">
                </div>
                <div class="plot-container">
                    <img src="cid:optimization_timeline" alt="Optimization Timeline">
                </div>
                """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_componentwise_summary(self, results: Dict[str, Any]) -> str:
        html = "<h2>Optimization Summary</h2>"
        
        component_results = results.get('component_results', {})
        total_trials = sum(comp.get('n_trials', 0) for comp in component_results.values())
        
        html += '<table class="summary-table">'
        html += "<tr><th>Component</th><th>Best Score</th><th>Trials</th><th>Time</th></tr>"
        
        for component in results.get('component_order', []):
            if component in component_results:
                comp_data = component_results[component]
                html += f"""
                <tr>
                    <td>{component.replace('_', ' ').title()}</td>
                    <td>{comp_data.get('best_score', 0):.4f}</td>
                    <td>{comp_data.get('n_trials', 0)}</td>
                    <td>{self._format_duration(comp_data.get('optimization_time', 0))}</td>
                </tr>
                """
        
        html += f"""
        <tr style="background-color: #e9ecef; font-weight: bold;">
            <td>TOTAL</td>
            <td>-</td>
            <td>{total_trials}</td>
            <td>{self._format_duration(results.get('optimization_time', 0))}</td>
        </tr>
        """
        html += "</table>"
        
        if 'early_stopped' in results and results['early_stopped']:
            html += '<p style="color: #ffc107; margin-top: 10px;"><strong>⚡ Early stopped - target achieved!</strong></p>'
        
        return html
    
    def _create_component_details(self, results: Dict[str, Any]) -> str:
        html = "<h2>Component Optimization Details</h2>"
        
        component_results = results.get('component_results', {})
        
        for component in results.get('component_order', []):
            if component not in component_results:
                continue
                
            comp_data = component_results[component]
            best_config = comp_data.get('best_config', {})
            
            html += f'<div class="component-card">'
            html += f'<div class="component-header">'
            html += f'<span class="component-name">{component.replace("_", " ").title()}</span>'
            html += f'<span class="component-score">Score: {comp_data.get("best_score", 0):.4f}</span>'
            html += '</div>'
            
            if best_config:
                html += '<div class="config-box">'
                html += '<strong>Best Configuration:</strong><br>'
                config_str = json.dumps(best_config, indent=2, cls=NumpyEncoder)
                html += config_str.replace('\n', '<br>').replace(' ', '&nbsp;')
                html += '</div>'
            
            html += f'<div style="margin-top: 10px;">'
            html += f'<span>Search Space Size: {comp_data.get("search_space_size", "N/A")}</span> | '
            html += f'<span>Trials: {comp_data.get("n_trials", 0)}</span>'
            html += '</div>'
            
            html += '</div>'
        
        return html
    
    def _attach_componentwise_plots(self, html_part: MIMEMultipart, results: Dict[str, Any]):
        component_results = results.get('component_results', {})
        component_order = results.get('component_order', [])
        
        if not component_order:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        components = []
        scores = []
        
        for comp in component_order:
            if comp in component_results:
                components.append(comp.replace('_', '\n'))
                scores.append(component_results[comp].get('best_score', 0))
        
        bars = ax.bar(range(len(components)), scores, color='#007bff', alpha=0.8)
        
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xticks(range(len(components)))
        ax.set_xticklabels(components, fontsize=10)
        ax.set_ylabel('Best Score', fontsize=12)
        ax.set_title('Component Optimization Results', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(scores) * 1.15 if scores else 1)
        ax.grid(axis='y', alpha=0.3)
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        
        img = MIMEBase('image', 'png')
        img.set_payload(img_buffer.read())
        encoders.encode_base64(img)
        img.add_header('Content-ID', '<component_scores>')
        img.add_header('Content-Disposition', 'inline')
        html_part.attach(img)
        
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        times = []
        labels = []
        
        for comp in component_order:
            if comp in component_results:
                comp_time = component_results[comp].get('optimization_time', 0)
                times.append(comp_time)
                labels.append(comp.replace('_', ' ').title())
        
        if times:
            positions = range(len(times))
            bars = ax.barh(positions, times, color='#28a745', alpha=0.8)
            
            for i, (bar, time) in enumerate(zip(bars, times)):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                       f'{self._format_duration(time)}', va='center', fontsize=10)
            
            ax.set_yticks(positions)
            ax.set_yticklabels(labels, fontsize=10)
            ax.set_xlabel('Optimization Time', fontsize=12)
            ax.set_title('Component Optimization Timeline', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        
        img_buffer2 = io.BytesIO()
        plt.savefig(img_buffer2, format='png', bbox_inches='tight', dpi=150)
        img_buffer2.seek(0)
        
        img2 = MIMEBase('image', 'png')
        img2.set_payload(img_buffer2.read())
        encoders.encode_base64(img2)
        img2.add_header('Content-ID', '<optimization_timeline>')
        img2.add_header('Content-Disposition', 'inline')
        html_part.attach(img2)
        
        plt.close()
    
    def _create_email_body(
        self,
        experiment_name: str,
        results: Dict[str, Any],
        duration: float,
        status: str,
        error_message: str
    ) -> str:
        body = f"Experiment: {experiment_name}\n"
        body += f"Status: {status.upper()}\n"
        body += f"Duration: {self._format_duration(duration)}\n"
        body += f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if status == "failed" and error_message:
            body += f"Error: {error_message}\n\n"
        
        if results and status == "completed":
            body += "Results Summary:\n"
            body += f"- Best Score: {results.get('best_score', 'N/A')}\n"
            body += f"- Best Latency: {results.get('best_latency', 'N/A')}s\n"
            body += f"- Total Trials: {results.get('total_trials', 'N/A')}\n"
            if 'pareto_front' in results:
                body += f"- Pareto Front Size: {len(results['pareto_front'])}\n"
        
        return body
    
    def _create_html_content(
        self,
        experiment_name: str,
        results: Dict[str, Any],
        duration: float,
        status: str,
        error_message: str,
        attach_plots: bool = True
    ) -> str:
        status_color = {
            "completed": "#28a745",
            "failed": "#dc3545",
            "early_stopped": "#ffc107"
        }
        color = status_color.get(status, "#17a2b8")
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: {color}; color: white; padding: 20px; border-radius: 5px; }}
                .content {{ margin-top: 20px; }}
                .results-table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                .results-table th, .results-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .results-table th {{ background-color: #f2f2f2; }}
                .metric-card {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .config-box {{ background-color: #e9ecef; padding: 10px; margin: 10px 0; border-radius: 5px; font-family: monospace; font-size: 12px; }}
                .plot-container {{ margin: 20px 0; text-align: center; }}
                .plot-container img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{experiment_name}</h1>
                <p>Status: <strong>{status.upper()}</strong> | Duration: {self._format_duration(duration)}</p>
                <p>Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="content">
        """
        
        if status == "failed" and error_message:
            html += f"""
                <div style="background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <strong>Error:</strong> {error_message}
                </div>
            """
        
        if results and status == "completed":
            html += self._create_results_section(results)
            
            if attach_plots:
                html += """
                <h2>Optimization Plots</h2>
                <div class="plot-container">
                    <img src="cid:optimization_history" alt="Optimization History">
                </div>
                <div class="plot-container">
                    <img src="cid:pareto_front" alt="Pareto Front">
                </div>
                """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_results_section(self, results: Dict[str, Any]) -> str:
        html = "<h2>Results Summary</h2>"
        
        html += '<div class="metric-card">'
        html += f"<h3>Performance Metrics</h3>"
        html += f"<p><strong>Best Score:</strong> {results.get('best_score', 'N/A'):.4f}</p>"
        html += f"<p><strong>Best Latency:</strong> {results.get('best_latency', 'N/A'):.2f}s</p>"
        html += f"<p><strong>Total Trials:</strong> {results.get('total_trials', 'N/A')}</p>"
        
        if results.get('early_stopped'):
            html += '<p style="color: #ffc107;"><strong>⚡ Early stopped - target achieved!</strong></p>'
        
        if 'pareto_front' in results:
            html += f"<p><strong>Pareto Front Size:</strong> {len(results['pareto_front'])}</p>"
        
        html += "</div>"
        
        if 'best_config' in results and isinstance(results['best_config'], dict):
            config = results['best_config'].get('config', results['best_config'])
            if isinstance(config, dict):
                html += '<div class="metric-card">'
                html += "<h3>Best Configuration</h3>"
                html += '<div class="config-box">'
                html += json.dumps(config, indent=2, cls=NumpyEncoder).replace('\n', '<br>').replace(' ', '&nbsp;')
                html += '</div>'
                html += "</div>"
        
        if 'pareto_front' in results and results['pareto_front']:
            html += self._create_pareto_table(results['pareto_front'][:5])
        
        return html
    
    def _create_pareto_table(self, pareto_front: List[Dict[str, Any]]) -> str:
        html = '<div class="metric-card">'
        html += "<h3>Top Pareto-Optimal Solutions</h3>"
        html += '<table class="results-table">'
        html += "<tr><th>Rank</th><th>Score</th><th>Latency (s)</th><th>Trial #</th></tr>"
        
        for i, solution in enumerate(pareto_front):
            html += f"""
            <tr>
                <td>{i+1}</td>
                <td>{solution.get('score', 'N/A'):.4f}</td>
                <td>{solution.get('latency', 'N/A'):.2f}</td>
                <td>{solution.get('trial_number', 'N/A')}</td>
            </tr>
            """
        
        html += "</table></div>"
        return html
    
    def _attach_inline_plots(self, html_part: MIMEMultipart, results: Dict[str, Any]):
        if 'all_trials' in results and results['all_trials']:
            trials_data = results['all_trials']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            trial_numbers = [t['trial_number'] for t in trials_data]
            scores = [t['score'] for t in trials_data]
            latencies = [t.get('latency', 0) for t in trials_data]
            
            ax1.plot(trial_numbers, scores, 'b-', alpha=0.7)
            ax1.scatter(trial_numbers, scores, c='blue', s=20)
            ax1.set_xlabel('Trial Number')
            ax1.set_ylabel('Score')
            ax1.set_title('Score Progression')
            ax1.grid(True, alpha=0.3)
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
            img_buffer.seek(0)
            
            img = MIMEBase('image', 'png')
            img.set_payload(img_buffer.read())
            encoders.encode_base64(img)
            img.add_header('Content-ID', '<optimization_history>')
            img.add_header('Content-Disposition', 'inline')
            html_part.attach(img)
            
            plt.close()
            
            if 'pareto_front' in results and results['pareto_front']:
                plt.figure(figsize=(8, 6))
                
                all_scores = [t['score'] for t in trials_data]
                all_latencies = [t.get('latency', 0) for t in trials_data]
                plt.scatter(all_scores, all_latencies, c='lightblue', alpha=0.5, s=30, label='All Trials')
                
                pf_scores = [t['score'] for t in results['pareto_front']]
                pf_latencies = [t['latency'] for t in results['pareto_front']]
                plt.scatter(pf_scores, pf_latencies, c='red', marker='x', s=100, label='Pareto Front')
                
                plt.xlabel('Score')
                plt.ylabel('Latency (s)')
                plt.title('Pareto Front Visualization')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                img_buffer2 = io.BytesIO()
                plt.savefig(img_buffer2, format='png', bbox_inches='tight', dpi=150)
                img_buffer2.seek(0)
                
                img2 = MIMEBase('image', 'png')
                img2.set_payload(img_buffer2.read())
                encoders.encode_base64(img2)
                img2.add_header('Content-ID', '<pareto_front>')
                img2.add_header('Content-Disposition', 'inline')
                html_part.attach(img2)
                
                plt.close()
    
    def _send_email(self, message: MIMEMultipart):
        context = ssl.create_default_context()
        
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls(context=context)
            server.login(self.sender_email, self.sender_password)
            server.send_message(message)
    
    def _format_duration(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


class ExperimentNotificationWrapper:
    def __init__(self, optimizer, email_notifier: ExperimentEmailNotifier):
        self.optimizer = optimizer
        self.email_notifier = email_notifier
        self.start_time = None
        self.experiment_name = None
    
    def run_with_notification(self, experiment_name: str = None) -> Dict[str, Any]:
        self.experiment_name = experiment_name or f"{self.optimizer.study_name}"
        self.start_time = datetime.now().timestamp()
        
        results = None
        try:
            results = self.optimizer.optimize()
            
            duration = datetime.now().timestamp() - self.start_time

            if hasattr(self.optimizer, 'early_stopping_threshold'):
                if self._check_early_stopped(results, self.optimizer.early_stopping_threshold):
                    status = "early_stopped"
                    results['early_stopped'] = True
                else:
                    status = "completed"
            else:
                status = "completed"

            try:
                self.email_notifier.send_experiment_notification(
                    experiment_name=self.experiment_name,
                    results=results,
                    duration=duration,
                    status=status,
                    attach_plots=True
                )
            except Exception as email_error:
                print(f"Warning: Failed to send email after successful optimization: {email_error}")
            
            return results
            
        except Exception as e:
            duration = datetime.now().timestamp() - self.start_time
            error_msg = str(e)
            
            try:
                self.email_notifier.send_experiment_notification(
                    experiment_name=self.experiment_name,
                    results=results, 
                    duration=duration,
                    status="failed",
                    error_message=error_msg,
                    attach_plots=False
                )
            except:
                pass  
            
            raise
    
    def _check_early_stopped(self, results: Dict[str, Any], threshold: float) -> bool:
        if 'component_results' in results:
            return any(
                comp.get('best_score', 0) >= threshold 
                for comp in results.get('component_results', {}).values()
            )
        elif 'best_config' in results:
            return results['best_config'].get('score', 0) >= threshold
        
        return False
        
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):  
            return int(obj)
        if isinstance(obj, np.floating):  
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)