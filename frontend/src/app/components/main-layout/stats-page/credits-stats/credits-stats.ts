import { Component } from '@angular/core';
import { TranslatePipe } from '@ngx-translate/core';

import { ReactiveFormsModule } from '@angular/forms';
import { InferenceActionsCreditsForm } from './inference-actions-credits-form/inference-actions-credits-form';
import { CronjobCreditsForm } from './cronjob-credits-form/cronjob-credits-form';

@Component({
  selector: 'app-credits-stats',
  imports: [TranslatePipe, ReactiveFormsModule, InferenceActionsCreditsForm, CronjobCreditsForm],
  templateUrl: './credits-stats.html',
  styleUrl: './credits-stats.css',
})
export class CreditsStats {}
