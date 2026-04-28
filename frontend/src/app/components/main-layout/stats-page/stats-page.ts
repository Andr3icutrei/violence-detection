import { Component } from '@angular/core';
import { NgxChartsModule } from '@swimlane/ngx-charts';
import { InspectStats } from './inspect-stats/inspect-stats';

@Component({
  selector: 'app-stats-page',
  imports: [NgxChartsModule, InspectStats],
  templateUrl: './stats-page.html',
  styleUrl: './stats-page.css',
})
export class StatsPage {}
