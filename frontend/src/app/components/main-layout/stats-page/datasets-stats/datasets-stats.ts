import { ChangeDetectorRef, Component, OnInit } from '@angular/core';
import { DatasetsService } from '../../../../services/datasets/datasets-service';
import { DatasetsStatsResponseDto } from '../../../../core/api/models/datasets-stats-response-dto';
import { Router } from '@angular/router';
import { I18nPluralPipe } from '@angular/common';
import { TranslatePipe } from '@ngx-translate/core';

@Component({
  selector: 'app-datasets-stats',
  imports: [I18nPluralPipe, TranslatePipe],
  templateUrl: './datasets-stats.html',
  styleUrl: './datasets-stats.css',
})
export class DatasetsStats implements OnInit {
  datasetsStats!: DatasetsStatsResponseDto;

  videoMapping: { [k: string]: string } = {
    '=0': '0 inferences',
    '=1': '1 inference',
    other: '# inferences',
  };

  constructor(
    private cdr: ChangeDetectorRef,
    private router: Router,
    private datasetsService: DatasetsService,
  ) {}

  ngOnInit(): void {
    this.loadDatasetsInformation();
  }

  private loadDatasetsInformation(): void {
    this.datasetsService.getDatasetsStats().subscribe({
      next: (data: DatasetsStatsResponseDto) => {
        this.datasetsStats = data;
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.router.navigate(['/dashboard']);
      },
    });
  }
}
