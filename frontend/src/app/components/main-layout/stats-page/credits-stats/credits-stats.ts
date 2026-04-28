import { ChangeDetectorRef, Component, OnInit } from '@angular/core';
import { TranslatePipe } from '@ngx-translate/core';
import { CreditsService } from '../../../../services/credits/credits.service';
import { Router } from '@angular/router';
import { InferenceActionsService } from '../../../../services/inference_actions/inference-actions.service';
import { InferenceActionResponseDto } from '../../../../core/api/models/inference-action-response-dto';

@Component({
  selector: 'app-credits-stats',
  imports: [TranslatePipe],
  templateUrl: './credits-stats.html',
  styleUrl: './credits-stats.css',
})
export class CreditsStats implements OnInit {
  creditsCronjobUpdate!: number;
  inferenceActionsStats!: InferenceActionResponseDto[];

  constructor(
    private cdr: ChangeDetectorRef,
    private router: Router,
    private creditsService: CreditsService,
    private inferenceActionsService: InferenceActionsService,
  ) {}

  ngOnInit(): void {
    this.loadCreditsInformation();
    this.loadInferenceActionsInformation();
  }

  private loadCreditsInformation(): void {
    this.creditsService.getCreditsCronjobUpdate().subscribe({
      next: (data: number) => {
        this.creditsCronjobUpdate = data;
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.router.navigate(['/dashboard']);
      },
    });
  }

  private loadInferenceActionsInformation(): void {
    this.inferenceActionsService.getInferenceActionsStats().subscribe({
      next: (data: InferenceActionResponseDto[]) => {
        this.inferenceActionsStats = data;
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.router.navigate(['/dashboard']);
      },
    });
  }
}
