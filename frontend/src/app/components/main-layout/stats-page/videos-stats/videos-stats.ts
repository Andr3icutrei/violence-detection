import { ChangeDetectorRef, Component, OnInit } from '@angular/core';
import { TranslatePipe } from '@ngx-translate/core';
import { Router } from '@angular/router';
import { InferenceHistoryService } from '../../../../services/inference_history/inference-history.service';
import { InferenceHistoryStatsResponseDto } from '../../../../core/api/models/inference-history-stats-response-dto';
import { DatasetsStatsResponseDto } from '../../../../core/api/models/datasets-stats-response-dto';
import { DatasetsService } from '../../../../services/datasets/datasets-service';
import { I18nPluralPipe } from '@angular/common';
import { InferenceHistoryClassificationStatsResponseDto } from '../../../../core/api/models/inference-history-classification-stats-response-dto';
import { InferenceHistoryPeopleTrackingStatsResponseDto } from '../../../../core/api/models/inference-history-people-tracking-stats-response-dto';
import { NgxChartsModule } from '@swimlane/ngx-charts';

@Component({
  selector: 'app-videos-stats',
  imports: [TranslatePipe, NgxChartsModule],
  templateUrl: './videos-stats.html',
  styleUrl: './videos-stats.css',
})
export class VideosStats implements OnInit {
  inferenceHistoryStats!: InferenceHistoryStatsResponseDto;

  selectedDate = new Date();

  colorScheme: any = {
    domain: ['#9c27b0'],
  };

  legend: boolean = false;
  xAxis: boolean = true;
  yAxis: boolean = true;
  showYAxisLabel: boolean = false;
  showXAxisLabel: boolean = false;
  xAxisLabel: string = `${this.selectedDate.toLocaleDateString('en', { month: 'long' })} ${this.selectedDate.getFullYear()}`;
  yAxisLabel: string = 'Classification runs';
  yAxisLabelPeopleTracking: string = 'People tracking runs';
  timeline: boolean = true;

  multiClassificationRuns: any[] = [];
  multiPeopleTrackingRuns: any[] = [];

  constructor(
    private cdr: ChangeDetectorRef,
    private router: Router,
    private inferenceHistoryService: InferenceHistoryService,
  ) {}

  ngOnInit(): void {
    this.loadInferenceHistoryInformation();
  }

  private loadInferenceHistoryInformation(): void {
    this.inferenceHistoryService.getInferenceHistoryStats().subscribe({
      next: (data: InferenceHistoryStatsResponseDto) => {
        this.inferenceHistoryStats = data;
        this.createMultis();
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.router.navigate(['/dashboard']);
      },
    });
  }

  private createMultis(): void {
    const year = this.selectedDate.getFullYear();
    const month = this.selectedDate.getMonth();

    this.xAxisLabel = `${this.selectedDate.toLocaleDateString('en', { month: 'long' })} ${year}`;

    const daysInMonth = new Date(year, month + 1, 0).getDate();
    const seriesDataClassification = [];
    const seriesDataPeopleTracking = [];

    for (let day = 1; day <= daysInMonth; day++) {
      const classificationRuns = this.inferenceHistoryStats.classification_runs.filter(
        (run: InferenceHistoryClassificationStatsResponseDto) => {
          const runDate = new Date(run.created_at);
          return (
            runDate.getFullYear() === year &&
            runDate.getMonth() === month &&
            runDate.getDate() === day
          );
        },
      );

      const peopleTrackingRuns = this.inferenceHistoryStats.people_tracking_runs.filter(
        (run: InferenceHistoryPeopleTrackingStatsResponseDto) => {
          const runDate = new Date(run.created_at);
          return (
            runDate.getFullYear() === year &&
            runDate.getMonth() === month &&
            runDate.getDate() === day
          );
        },
      );

      const classificationCount = classificationRuns.length;
      const peopleTrackingCount = peopleTrackingRuns.length;

      seriesDataClassification.push({
        name: day.toString(),
        value: classificationCount,
      });

      seriesDataPeopleTracking.push({
        name: day.toString(),
        value: peopleTrackingCount,
      });
    }

    this.multiClassificationRuns = [
      {
        name: 'Classification runs',
        series: seriesDataClassification,
      },
    ];
    this.multiPeopleTrackingRuns = [
      {
        name: 'People tracking runs',
        series: seriesDataPeopleTracking,
      },
    ];
  }
}
