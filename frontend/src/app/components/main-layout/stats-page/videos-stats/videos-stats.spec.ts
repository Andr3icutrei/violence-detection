import { ComponentFixture, TestBed } from '@angular/core/testing';

import { VideosStats } from './videos-stats';

describe('VideosStats', () => {
  let component: VideosStats;
  let fixture: ComponentFixture<VideosStats>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [VideosStats],
    }).compileComponents();

    fixture = TestBed.createComponent(VideosStats);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
