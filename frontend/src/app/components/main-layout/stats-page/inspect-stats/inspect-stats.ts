import { UsersStats } from '../users-stats/users-stats';
import { VideosStats } from '../videos-stats/videos-stats';
import { CreditsStats } from '../credits-stats/credits-stats';
import { Component } from '@angular/core';
import { DatasetsStats } from '../datasets-stats/datasets-stats';

@Component({
  selector: 'app-inspect-stats',
  imports: [UsersStats, CreditsStats, DatasetsStats, VideosStats],
  templateUrl: './inspect-stats.html',
  styleUrl: './inspect-stats.css',
})
export class InspectStats {}
