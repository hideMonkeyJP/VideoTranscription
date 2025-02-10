def save_results(self, output_dir: str, transcription: List[Dict[str, Any]], 
                    screenshots: List[Dict[str, Any]], analysis: Dict[str, Any]) -> Dict[str, str]:
        """処理結果を保存します"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # スクリーンショットの保存
            screenshots_dir = os.path.join(output_dir, f'screenshots_{timestamp}')
            os.makedirs(screenshots_dir, exist_ok=True)
            
            saved_screenshots = []
            for i, screenshot in enumerate(screenshots):
                if 'image' not in screenshot:
                    continue
                    
                # 画像の保存
                image_path = os.path.join(screenshots_dir, f'screenshot_{i:03d}.jpg')
                try:
                    screenshot['image'].save(image_path, 'JPEG')
                    
                    # 保存した画像の情報を記録
                    saved_screenshot = {
                        'path': os.path.relpath(image_path, output_dir),
                        'timestamp': screenshot['timestamp'],
                        'frame_number': screenshot.get('frame_number', i),
                        'importance_score': screenshot.get('importance_score', 0.0)
                    }
                    
                    if 'text' in screenshot:
                        saved_screenshot['text'] = screenshot['text']
                        
                    saved_screenshots.append(saved_screenshot)
                    
                except Exception as e:
                    self.logger.log_warning(f"スクリーンショットの保存に失敗: {str(e)}")
            
            # 最終結果の構築
            final_result = {
                'metadata': {
                    'timestamp': timestamp,
                    'version': '1.0',
                    'processing_time': round(time.time() - self.start_time, 2)
                },
                'transcription': transcription,
                'screenshots': saved_screenshots,
                'analysis': analysis
            }
            
            # 結果をJSONとして保存
            json_path = os.path.join(output_dir, f'result_{timestamp}.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            
            # HTMLレポートの生成
            html_path = os.path.join(output_dir, f'report_{timestamp}.html')
            self._generate_html_report(final_result, html_path)
            
            return {
                'json_path': json_path,
                'html_path': html_path,
                'screenshots_dir': screenshots_dir
            }
            
        except Exception as e:
            error_context = create_error_context(
                "result_saving",
                {
                    'output_dir': output_dir,
                    'transcription_length': len(transcription),
                    'screenshot_count': len(screenshots)
                }
            )
            self.logger.log_error(e, error_context)
            raise OutputError("結果の保存に失敗しました", error_context)

    def sync_to_notion(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """分析結果をNotionと同期します"""
        try:
            if not self.notion_client:
                self.logger.logger.warning("Notion連携が無効です")
                return None
                
            self.logger.logger.info("Notionとの同期を開始します")
            
            # 設定の取得
            notion_config = self.config.get('notion', {})
            
            # ページのプロパティを準備
            properties = {
                'タイトル': {
                    'title': [{
                        'text': {
                            'content': f"動画分析レポート {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                        }
                    }]
                },
                'ステータス': {
                    'select': {
                        'name': '分析完了'
                    }
                },
                '作成日時': {
                    'date': {
                        'start': datetime.now().isoformat()
                    }
                }
            }
            
            # キーワードとトピックの追加
            if 'keywords' in analysis:
                properties['キーワード'] = {
                    'multi_select': [
                        {'name': keyword} for keyword in analysis['keywords'][:10]
                    ]
                }
                
            if 'topics' in analysis:
                properties['トピック'] = {
                    'rich_text': [{
                        'text': {
                            'content': '\n• ' + '\n• '.join(analysis['topics'][:5])
                        }
                    }]
                }
                
            # シーン分析の追加
            if 'scene_analyses' in analysis:
                scene_content = []
                for scene in analysis['scene_analyses']:
                    timestamp = scene.get('timestamp', 0)
                    summary = scene.get('summary', {})
                    
                    scene_text = f"### {int(timestamp)}秒\n"
                    
                    if 'main_points' in summary:
                        scene_text += "\n主要ポイント:\n"
                        for point in summary['main_points']:
                            scene_text += f"• {point}\n"
                            
                    scene_content.append(scene_text)
                    
                if scene_content:
                    properties['シーン分析'] = {
                        'rich_text': [{
                            'text': {
                                'content': '\n'.join(scene_content)
                            }
                        }]
                    }
                    
            # ページの作成
            try:
                page = self.notion_client.create_page(properties)
                self.logger.logger.info(f"Notionページを作成しました: {page.get('id')}")
                return page
                
            except Exception as e:
                self.logger.logger.error(f"Notionページの作成に失敗: {str(e)}")
                return None
                
        except Exception as e:
            error_context = create_error_context(
                "notion_sync",
                {'analysis_keys': list(analysis.keys())}
            )
            self.logger.log_error(e, error_context)
            return None

    def _generate_html_report(self, result: Dict[str, Any], output_path: str):
        """HTML形式のレポートを生成します"""
        try:
            # テンプレートの読み込み
            template_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'templates',
                'notion.html'
            )
            
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()
            
            # データの準備
            metadata = result['metadata']
            transcription = result['transcription']
            screenshots = result['screenshots']
            analysis = result['analysis']
            
            # スクリーンショットのHTMLを生成
            screenshot_html = []
            for screenshot in screenshots:
                screenshot_html.append(f"""
                <div class="screenshot-item">
                    <img src="{screenshot['path']}" alt="Screenshot at {screenshot['timestamp']}s">
                    <div class="screenshot-info">
                        <p>Timestamp: {screenshot['timestamp']}s</p>
                        {f"<p>Text: {html.escape(screenshot['text'])}</p>" if 'text' in screenshot else ''}
                    </div>
                </div>
                """)
            
            # トランスクリプションのHTMLを生成
            transcript_html = []
            for segment in transcription:
                transcript_html.append(f"""
                <div class="transcript-segment">
                    <span class="timestamp">{segment['start']}s - {segment['end']}s</span>
                    <p>{html.escape(segment['text'])}</p>
                </div>
                """)
            
            # 分析結果のHTMLを生成
            analysis_html = []
            if 'keywords' in analysis:
                analysis_html.append("<h3>キーワード</h3>")
                analysis_html.append("<ul>")
                for keyword in analysis['keywords']:
                    analysis_html.append(f"<li>{html.escape(keyword)}</li>")
                analysis_html.append("</ul>")
            
            if 'topics' in analysis:
                analysis_html.append("<h3>トピック</h3>")
                analysis_html.append("<ul>")
                for topic in analysis['topics']:
                    analysis_html.append(f"<li>{html.escape(topic)}</li>")
                analysis_html.append("</ul>")
            
            # テンプレートの置換
            html_content = template.replace(
                "{{timestamp}}", metadata['timestamp']
            ).replace(
                "{{processing_time}}", str(metadata['processing_time'])
            ).replace(
                "{{screenshots}}", "\n".join(screenshot_html)
            ).replace(
                "{{transcript}}", "\n".join(transcript_html)
            ).replace(
                "{{analysis}}", "\n".join(analysis_html)
            )
            
            # HTMLファイルの保存
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
        except Exception as e:
            self.logger.log_error(e, {'operation': 'html_report_generation'})
            raise OutputError("HTMLレポートの生成に失敗しました")
