import { UsersStats } from '../users-stats/users-stats';
import { CreditsStats } from '../credits-stats/credits-stats';
import { Component } from '@angular/core';
import { DatasetsStats } from '../datasets-stats/datasets-stats';
import { InferenceStats } from '../inference-stats/inference-stats';

@Component({
  selector: 'app-inspect-stats',
  imports: [UsersStats, CreditsStats, DatasetsStats, InferenceStats],
  templateUrl: './inspect-stats.html',
  styleUrl: './inspect-stats.css',
})
export class InspectStats {}
